#!/usr/bin/env python3
"""
phase_04_inference.py
Phase 4 Execution Script: Constrained Decoding (Outlines) & End-to-End Evaluation.

Supports two execution modes:
1. 'eval' (default): Loads Phase 2 verified .pt batches (e.g. dev/), runs the trained
   Phase 3 HybridRiskModel from models/lora_adapters/, applies Phase 4 Outlines
   constrained decoding for strict 1-5 integer outputs, and computes MAE,
   Exact Match Accuracy, Within-1 Accuracy, and a 5x5 Confusion Matrix.
2. 'predict': Executes the complete 4-Phase Pipeline end-to-end on raw JSON/JSONL files
   (Phase 1 DAG -> Phase 2 LangGraph -> Phase 3 HybridRiskModel -> Phase 4 Outlines)
   and exports SemEval submission artifacts (predictions.jsonl & submission.csv).
"""

import argparse
import csv
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Ensure src and project root are on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent_risk.dag_parser import TrajectoryDAG
from agent_risk.models.constrained_decoder import ConstrainedRiskDecoder
from agent_risk.models.hybrid_llama import HybridRiskModel
from agent_risk.simulator.state import SimulationState
from agent_risk.simulator.verifier_nodes import verify_trajectory
from experiments.phase_01_preprocessing import stream_file_records
from experiments.phase_03_hybrid_gnn import HybridCollateFn, VerifiedTrajectoryDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase04ConstrainedInference")


class InMemoryTrajectoryDataset(Dataset):
    """Dataset wrapper for raw trajectories processed on-the-fly through Phases 1 & 2."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.samples = records

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def preprocess_raw_to_phase2_records(raw_path: Path) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Executes Phase 1 (TrajectoryDAG) + Phase 2 (LangGraph) sequentially on a raw
    JSON/JSONL file. Returns (records, has_ground_truth_labels).
    """
    from langgraph.graph import END, START, StateGraph

    builder = StateGraph(SimulationState)
    builder.add_node("verify_trajectory", verify_trajectory)
    builder.add_edge(START, "verify_trajectory")
    builder.add_edge("verify_trajectory", END)
    verifier_app = builder.compile()

    records: List[Dict[str, Any]] = []
    has_labels = False

    for raw_item in stream_file_records(raw_path):
        if "risk_severity" in raw_item or "risk_level" in raw_item or "risk" in raw_item:
            has_labels = True

        # Phase 1: DAG construction & error branch pruning
        dag = TrajectoryDAG(raw_item, scale_to_1_5=True)
        dag_dict = dag.to_dict()

        # Phase 2: LangGraph deterministic hallucination check
        ver_state = verifier_app.invoke({
            "trajectory_id": dag_dict["id"],
            "tokenized_text": dag_dict["tokenized_text"],
            "hallucination_flag": False,
            "risk_modifier": 0.0,
        })

        records.append({
            "id": dag_dict["id"],
            "tokenized_text": dag_dict["tokenized_text"],
            "adj_matrix": torch.tensor(dag_dict["adjacency_matrix"], dtype=torch.float32),
            "risk_level": float(dag_dict["risk_level"]),
            "hallucination_flag": 1.0 if ver_state["hallucination_flag"] else 0.0,
            "risk_modifier": float(ver_state["risk_modifier"]),
        })

    logger.info(f"Preprocessed {len(records)} raw trajectories through Phases 1 & 2.")
    return records, has_labels


def compute_evaluation_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Computes continuous MAE, strict 1-5 integer MAE (SemEval official metric),
    exact match accuracy, off-by-1 accuracy, per-class MAE, and a 5x5 confusion matrix.
    """
    n = len(records)
    if n == 0:
        return {"num_samples": 0}

    cont_errors = [abs(r["continuous_score"] - r["true_risk"]) for r in records]
    int_errors = [abs(r["predicted_risk"] - int(round(r["true_risk"]))) for r in records]
    exact_matches = [1 if r["predicted_risk"] == int(round(r["true_risk"])) else 0 for r in records]
    within_1 = [1 if e <= 1 else 0 for e in int_errors]

    # 5x5 confusion matrix: rows = true (1..5), cols = pred (1..5)
    confusion = [[0] * 5 for _ in range(5)]
    per_class: Dict[str, Dict[str, Any]] = {
        str(k): {"count": 0, "int_mae": 0.0, "exact_acc": 0.0} for k in range(1, 6)
    }
    class_errors: Dict[int, List[int]] = {k: [] for k in range(1, 6)}
    class_exact: Dict[int, List[int]] = {k: [] for k in range(1, 6)}

    for r in records:
        t_int = max(1, min(5, int(round(r["true_risk"]))))
        p_int = max(1, min(5, int(r["predicted_risk"])))
        confusion[t_int - 1][p_int - 1] += 1
        class_errors[t_int].append(abs(p_int - t_int))
        class_exact[t_int].append(1 if p_int == t_int else 0)

    for k in range(1, 6):
        cnt = len(class_errors[k])
        per_class[str(k)] = {
            "count": cnt,
            "int_mae": round(sum(class_errors[k]) / cnt, 4) if cnt > 0 else None,
            "exact_acc_pct": round(100.0 * sum(class_exact[k]) / cnt, 2) if cnt > 0 else None,
        }

    return {
        "num_samples": n,
        "continuous_mae": round(sum(cont_errors) / n, 4),
        "integer_mae": round(sum(int_errors) / n, 4),
        "exact_match_accuracy_pct": round(100.0 * sum(exact_matches) / n, 2),
        "within_1_accuracy_pct": round(100.0 * sum(within_1) / n, 2),
        "per_class_metrics": per_class,
        "confusion_matrix_1_to_5": confusion,
    }


def load_models_for_phase4(
    adapter_dir: Path,
    device_map: Optional[str],
    load_in_4bit: bool,
    lm_weight: float,
    use_pure_outlines: bool,
    hf_token: Optional[str],
) -> Tuple[HybridRiskModel, ConstrainedRiskDecoder, Any, torch.device]:
    """
    Loads the trained Phase 3 HybridRiskModel and Phase 4 ConstrainedRiskDecoder.
    When constrained LM logit scoring is enabled (lm_weight > 0 or use_pure_outlines),
    loads AutoModelForCausalLM once and shares its .model backbone in VRAM.
    """
    has_cuda = torch.cuda.is_available()
    head_device = torch.device("cuda:0" if has_cuda else "cpu")
    torch_dtype = torch.float16 if has_cuda else torch.float32
    resolved_device_map = None if (not has_cuda or (device_map or "").lower() == "none") else device_map

    fusion_path = adapter_dir / "fusion_heads.pt"
    if not fusion_path.exists():
        raise FileNotFoundError(f"Could not find fusion_heads.pt in {adapter_dir}")
    fusion_state = torch.load(fusion_path, map_location="cpu", weights_only=False)
    model_name = fusion_state.get("model_name_or_path", "meta-llama/Meta-Llama-3-8B")

    logger.info(f"Loading Tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|pad|>"
    tokenizer.truncation_side = "left"

    needs_causal_lm = (lm_weight > 0.0) or use_pure_outlines
    causal_lm = None

    if needs_causal_lm:
        logger.info(
            f"Loading AutoModelForCausalLM ({model_name}) in shared-VRAM mode "
            f"(4bit={load_in_4bit}, device_map={resolved_device_map})..."
        )
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if resolved_device_map is not None:
            load_kwargs["device_map"] = resolved_device_map
        if hf_token is not None:
            load_kwargs["token"] = hf_token
        if load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        causal_lm = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        causal_lm.eval()

        hybrid_model = HybridRiskModel.from_adapters(
            adapter_dir=adapter_dir,
            base_model=causal_lm.model,
            torch_dtype=torch_dtype,
            device_map=resolved_device_map,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
        )
    else:
        logger.info(
            f"Loading HybridRiskModel directly from {adapter_dir} "
            f"(4bit={load_in_4bit}, device_map={resolved_device_map})..."
        )
        hybrid_model = HybridRiskModel.from_adapters(
            adapter_dir=adapter_dir,
            torch_dtype=torch_dtype,
            device_map=resolved_device_map,
            load_in_4bit=load_in_4bit,
            hf_token=hf_token,
        )

    if resolved_device_map is not None:
        hybrid_model.gcn.to(head_device)
        hybrid_model.regression_head.to(head_device)
    else:
        hybrid_model.to(head_device)
        if causal_lm is not None:
            causal_lm.to(head_device)

    hybrid_model.eval()

    decoder = ConstrainedRiskDecoder(
        causal_lm=causal_lm,
        tokenizer=tokenizer,
        hybrid_weight=1.0,
        lm_weight=lm_weight,
    )
    return hybrid_model, decoder, tokenizer, head_device


def run_inference_loop(
    hybrid_model: HybridRiskModel,
    decoder: ConstrainedRiskDecoder,
    dataset: Dataset,
    tokenizer: Any,
    head_device: torch.device,
    micro_batch_size: int = 2,
    max_seq_len: int = 1024,
    use_pure_outlines: bool = False,
) -> List[Dict[str, Any]]:
    """
    Runs Phase 3 continuous prediction followed by Phase 4 Outlines-constrained
    strict 1-5 integer decoding across all samples in dataset.
    """
    collate_fn = HybridCollateFn(tokenizer=tokenizer, max_seq_len=max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    use_cuda_amp = (head_device.type == "cuda")
    results: List[Dict[str, Any]] = []
    sample_cursor = 0

    with torch.no_grad():
        for step, batch in enumerate(loader, start=1):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            adj_matrices = batch["adj_matrices"]
            risk_modifiers = batch["risk_modifiers"].to(head_device)
            targets = batch["risk_levels"]
            sample_ids = batch["sample_ids"]

            with torch.autocast(device_type=head_device.type, dtype=torch.float16, enabled=use_cuda_amp):
                cont_preds = hybrid_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adj_matrices=adj_matrices,
                    risk_modifiers=risk_modifiers,
                )

            cont_list = cont_preds.detach().cpu().tolist()
            mod_list = risk_modifiers.detach().cpu().tolist()
            targ_list = targets.detach().cpu().tolist()

            for i in range(len(sample_ids)):
                raw_sample = dataset[sample_cursor]
                sample_cursor += 1
                cont_score = float(cont_list[i])

                pred_int, diag = decoder.decode_single(
                    hybrid_score=cont_score,
                    tokenized_text=raw_sample.get("tokenized_text", ""),
                    hallucination_flag=float(raw_sample.get("hallucination_flag", 0.0)),
                    risk_modifier=float(mod_list[i]),
                    use_pure_outlines=use_pure_outlines,
                )

                # Hard guarantee: strict 1-5 integer output
                assert isinstance(pred_int, int) and 1 <= pred_int <= 5, (
                    f"Out-of-range risk level emitted: {pred_int}"
                )

                results.append({
                    "example_id": sample_ids[i],
                    "predicted_risk": pred_int,
                    "continuous_score": round(cont_score, 4),
                    "true_risk": float(targ_list[i]),
                    "hallucination_flag": float(raw_sample.get("hallucination_flag", 0.0)),
                    "risk_modifier": round(float(mod_list[i]), 3),
                    "decoder_backend": diag.get("backend", "unknown"),
                })

            if step % 20 == 0 or step == len(loader):
                logger.info(f"Processed batch [{step}/{len(loader)}] ({len(results)}/{len(dataset)} samples)")

    return results


def export_predictions(
    results: List[Dict[str, Any]],
    output_dir: Path,
    metrics: Optional[Dict[str, Any]],
    prefix: str,
) -> None:
    """Writes JSONL predictions, SemEval CSV submission, and evaluation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / f"{prefix}_predictions.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    csv_path = output_dir / f"{prefix}_submission.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["example_id", "risk_severity"])
        for r in results:
            writer.writerow([r["example_id"], r["predicted_risk"]])

    if metrics is not None:
        report_path = output_dir / f"{prefix}_eval_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation report saved -> {report_path}")

    logger.info(f"Predictions saved       -> {jsonl_path}")
    logger.info(f"Submission CSV saved    -> {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4: Outlines Constrained 1-5 Integer Decoding & Evaluation."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "predict"],
        default="eval",
        help="'eval' evaluates Phase 2 .pt batches; 'predict' runs full 4-phase pipeline on raw JSON/JSONL.",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "lora_adapters"),
        help="Path to Phase 3 saved adapters and fusion_heads.pt.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_2_verified" / "dev"),
        help="Directory of Phase 2 .pt batches (eval mode) or raw .json/.jsonl file (predict mode).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory to write predictions, submission CSV, and eval report.",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Accelerate device_map (e.g. 'auto' on Kaggle 2x T4, or 'none').",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load Llama 3 in 4-bit NF4 quantization (recommended on T4 GPUs).",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=2,
        help="Batch size during inference (default: 2).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum sequence length with left-truncation (default: 1024).",
    )
    parser.add_argument(
        "--lm_weight",
        type=float,
        default=0.05,
        help="Weight of constrained Outlines LM token logits (0.0 skips CausalLM generation for speed).",
    )
    parser.add_argument(
        "--use_pure_outlines",
        action="store_true",
        help="Ablation: ignore HybridRiskModel score and generate purely via Outlines regex [1-5].",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face access token (defaults to HF_TOKEN environment variable).",
    )

    args = parser.parse_args()
    start_time = time.time()

    data_path = Path(args.data_path)
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    # 1. Load dataset
    has_ground_truth = True
    if args.mode == "eval":
        dataset: Dataset = VerifiedTrajectoryDataset(data_path)
        prefix = f"phase4_{data_path.name}"
    else:
        records, has_ground_truth = preprocess_raw_to_phase2_records(data_path)
        dataset = InMemoryTrajectoryDataset(records)
        prefix = f"phase4_{data_path.stem}"

    if len(dataset) == 0:
        raise RuntimeError(f"No samples found at {data_path}")

    # 2. Load Phase 3 HybridRiskModel + Phase 4 ConstrainedRiskDecoder
    hybrid_model, decoder, tokenizer, head_device = load_models_for_phase4(
        adapter_dir=adapter_dir,
        device_map=args.device_map,
        load_in_4bit=args.load_in_4bit,
        lm_weight=args.lm_weight,
        use_pure_outlines=args.use_pure_outlines,
        hf_token=args.hf_token,
    )

    # 3. Execute Inference + Constrained Decoding
    logger.info(f"Running Phase 4 inference on {len(dataset)} samples (mode={args.mode})...")
    results = run_inference_loop(
        hybrid_model=hybrid_model,
        decoder=decoder,
        dataset=dataset,
        tokenizer=tokenizer,
        head_device=head_device,
        micro_batch_size=args.micro_batch_size,
        max_seq_len=args.max_seq_len,
        use_pure_outlines=args.use_pure_outlines,
    )

    # 4. Compute Metrics (if ground truth is available)
    metrics = compute_evaluation_metrics(results) if has_ground_truth else None
    export_predictions(results, output_dir, metrics, prefix=prefix)

    elapsed = time.time() - start_time
    logger.info("=" * 65)
    logger.info(f"Phase 4 Constrained Inference Complete in {elapsed:.1f}s!")
    if metrics is not None:
        logger.info(f"  Samples Evaluated         : {metrics['num_samples']}")
        logger.info(f"  Continuous Hybrid MAE     : {metrics['continuous_mae']:.4f}")
        logger.info(f"  Strict Integer (1-5) MAE  : {metrics['integer_mae']:.4f}")
        logger.info(f"  Exact Match Accuracy      : {metrics['exact_match_accuracy_pct']:.2f}%")
        logger.info(f"  Within-1 Accuracy (±1)    : {metrics['within_1_accuracy_pct']:.2f}%")
    logger.info("=" * 65)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


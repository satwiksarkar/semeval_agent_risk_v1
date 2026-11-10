#!/usr/bin/env python3
"""
phase_02_langgraph.py
Phase 2 Execution Script: LangGraph Deterministic Action Verifier & Hallucination Detector.

Evaluates Phase 1 trajectory DAGs for tool hallucinations (dangling calls and ignored errors)
before downstream PyTorch modeling.

Hardware-Optimized for Core i3 and 8GB RAM:
- Loads exactly one .pt batch into memory at a time.
- Executes verification through a compiled LangGraph state machine.
- Appends hallucination_flags and risk_modifiers as 1D PyTorch float32 tensors.
- Persists enriched batches to data/processed/phase_2_verified/ and immediately invokes gc.collect().
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List

import torch

# Filter user warnings from legacy pydantic shims if any
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph.graph import StateGraph, START, END
from agent_risk.simulator.state import SimulationState
from agent_risk.simulator.verifier_nodes import verify_trajectory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase02LangGraph")


def get_memory_mb() -> float:
    """Returns current process memory consumption in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def build_langgraph_verifier():
    """
    Constructs and compiles the deterministic LangGraph state machine
    for trajectory hallucination and risk verification.
    """
    graph_builder = StateGraph(SimulationState)
    graph_builder.add_node("verify_trajectory", verify_trajectory)
    graph_builder.add_edge(START, "verify_trajectory")
    graph_builder.add_edge("verify_trajectory", END)
    return graph_builder.compile()


def process_batch(
    batch_file: Path,
    output_dir: Path,
    verifier_app: Any,
) -> Dict[str, Any]:
    """
    Loads one batch .pt file, runs each tokenized sequence through the LangGraph verifier,
    appends tensor features, and saves the enriched batch to disk.
    """
    logger.info(f"Loading {batch_file.name} for LangGraph verification...")
    batch_data: Dict[str, Any] = torch.load(batch_file)

    tokenized_texts: List[str] = batch_data.get("tokenized_texts", [])
    sample_ids: List[str] = batch_data.get("sample_ids", [])
    num_samples = len(tokenized_texts)

    hallucination_flags: List[float] = []
    risk_modifiers: List[float] = []

    start_time = time.time()
    for idx, text in enumerate(tokenized_texts):
        sid = sample_ids[idx] if idx < len(sample_ids) else f"sample-{idx}"
        initial_state: SimulationState = {
            "trajectory_id": sid,
            "tokenized_text": text,
            "hallucination_flag": False,
            "risk_modifier": 0.0,
        }

        # Run through LangGraph
        result = verifier_app.invoke(initial_state)

        flag_val = 1.0 if result.get("hallucination_flag", False) else 0.0
        mod_val = float(result.get("risk_modifier", 0.0))

        hallucination_flags.append(flag_val)
        risk_modifiers.append(mod_val)

    # Convert to 1D PyTorch float32 tensors
    flags_tensor = torch.tensor(hallucination_flags, dtype=torch.float32)
    modifiers_tensor = torch.tensor(risk_modifiers, dtype=torch.float32)

    # Append new features to batch payload
    batch_data["hallucination_flags"] = flags_tensor
    batch_data["risk_modifiers"] = modifiers_tensor

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / batch_file.name
    torch.save(batch_data, out_file)

    elapsed = time.time() - start_time
    total_hallucinations = int(flags_tensor.sum().item())
    avg_modifier = float(modifiers_tensor.mean().item())

    stats = {
        "batch_file": batch_file.name,
        "num_samples": num_samples,
        "hallucinations_detected": total_hallucinations,
        "hallucination_rate": round((total_hallucinations / num_samples) * 100, 2) if num_samples > 0 else 0.0,
        "avg_risk_modifier": round(avg_modifier, 4),
        "flags_tensor_shape": list(flags_tensor.shape),
        "modifiers_tensor_shape": list(modifiers_tensor.shape),
        "elapsed_seconds": round(elapsed, 3),
        "memory_mb": round(get_memory_mb(), 1),
    }

    logger.info(
        f"Saved Verified Batch -> {out_file.name} | "
        f"Samples: {num_samples} | "
        f"Hallucinations: {total_hallucinations} ({stats['hallucination_rate']}%) | "
        f"Shapes: flags={list(flags_tensor.shape)}, modifiers={list(modifiers_tensor.shape)} | "
        f"RAM: {stats['memory_mb']} MB"
    )

    # Free memory immediately
    del batch_data, flags_tensor, modifiers_tensor, hallucination_flags, risk_modifiers
    gc.collect()

    return stats


def run_phase_02(
    input_base_dir: Path,
    output_base_dir: Path,
) -> Dict[str, Any]:
    """
    Orchestrates Phase 2 LangGraph verification across all available splits.
    """
    logger.info("Initializing LangGraph Trajectory Verifier state machine...")
    verifier_app = build_langgraph_verifier()
    logger.info("LangGraph compiled successfully!")

    manifest: Dict[str, Any] = {
        "task": "SemEval 2027 Task 10 (AgentRisk) - Phase 2",
        "description": "LangGraph Deterministic Tool Hallucination & Risk Modifier Enriched Batches",
        "splits": {},
    }

    splits = ["train", "dev"]
    found_any = False

    for split in splits:
        split_in_dir = input_base_dir / split
        split_out_dir = output_base_dir / split

        if not split_in_dir.exists():
            continue

        batch_files = sorted(split_in_dir.glob("batch_*.pt"))
        if not batch_files:
            continue

        found_any = True
        logger.info(f"[{split.upper()}] Processing {len(batch_files)} batch file(s)...")

        split_stats: List[Dict[str, Any]] = []
        split_total_samples = 0
        split_total_hallucinations = 0

        for b_file in batch_files:
            b_stat = process_batch(b_file, split_out_dir, verifier_app)
            split_stats.append(b_stat)
            split_total_samples += b_stat["num_samples"]
            split_total_hallucinations += b_stat["hallucinations_detected"]

        manifest["splits"][split] = {
            "total_samples": split_total_samples,
            "total_hallucinations": split_total_hallucinations,
            "hallucination_rate": round(
                (split_total_hallucinations / split_total_samples) * 100, 2
            )
            if split_total_samples > 0
            else 0.0,
            "batches": split_stats,
        }

    # If batches were stored directly in input_base_dir (non-split format)
    if not found_any:
        root_batch_files = sorted(input_base_dir.glob("batch_*.pt"))
        if root_batch_files:
            logger.info(f"[ROOT] Processing {len(root_batch_files)} batch file(s)...")
            split_stats = []
            for b_file in root_batch_files:
                b_stat = process_batch(b_file, output_base_dir, verifier_app)
                split_stats.append(b_stat)
            manifest["splits"]["default"] = {
                "batches": split_stats,
            }

    # Save Phase 2 Manifest
    output_base_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_base_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info("=" * 65)
    logger.info("Phase 2 LangGraph Verification Complete!")
    for s_name, s_data in manifest["splits"].items():
        logger.info(
            f"  Split '{s_name}': {s_data.get('total_samples', 0)} samples | "
            f"{s_data.get('total_hallucinations', 0)} hallucinations ({s_data.get('hallucination_rate', 0)}%)"
        )
    logger.info(f"Verified dataset manifest saved to: {manifest_path.name}")
    logger.info("=" * 65)

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: LangGraph Action Verifier and Hallucination Filter."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_1_dag"),
        help="Input directory containing Phase 1 .pt batches (default: data/processed/phase_1_dag/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_2_verified"),
        help="Output directory for enriched Phase 2 .pt batches (default: data/processed/phase_2_verified/)",
    )

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    run_phase_02(input_dir, output_dir)


if __name__ == "__main__":
    main()


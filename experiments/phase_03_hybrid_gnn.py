#!/usr/bin/env python3
"""
phase_03_hybrid_gnn.py
Phase 3 Execution Script: Kaggle-Optimized Training Loop for Hybrid Llama 3 + GNN.

Fuses LoRA-adapted Llama 3 representations with VanillaGCN DAG embeddings
using Mean Absolute Error (MAE / L1Loss) and physical risk modifier penalties.

Hardware Optimization for Kaggle 2x T4 GPUs (16GB VRAM each):
- Micro-batch size of 2 with gradient accumulation steps of 8 (effective batch size: 16).
- Mixed precision training with torch.autocast (float16) and GradScaler.
- LoRA adapters (r=8, alpha=16) to freeze base LLM weights and save VRAM.
- Persists best adapter checkpoint to models/lora_adapters/ based on validation MAE.
"""

import argparse
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
from transformers import AutoTokenizer, PreTrainedTokenizer

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_risk.models.hybrid_llama import HybridRiskModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase03HybridTraining")


class VerifiedTrajectoryDataset(Dataset):
    """
    Loads Phase 2 verified .pt batch files from disk into an indexed dataset.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.samples: List[Dict[str, Any]] = []

        batch_files = sorted(data_dir.glob("batch_*.pt"))
        if not batch_files:
            logger.warning(f"No batch_*.pt files found in {data_dir}")
            return

        for b_file in batch_files:
            logger.info(f"Loading batch file: {b_file.name}")
            data = torch.load(b_file)
            num_samples = data.get("num_samples", len(data.get("tokenized_texts", [])))
            for i in range(num_samples):
                self.samples.append({
                    "id": data["sample_ids"][i] if "sample_ids" in data else f"sample-{i}",
                    "tokenized_text": data["tokenized_texts"][i],
                    "adj_matrix": data["adj_matrices"][i],
                    "risk_level": float(data["risk_levels"][i].item() if torch.is_tensor(data["risk_levels"][i]) else data["risk_levels"][i]),
                    "risk_modifier": float(data["risk_modifiers"][i].item() if torch.is_tensor(data["risk_modifiers"][i]) else data["risk_modifiers"][i]),
                    "hallucination_flag": float(data["hallucination_flags"][i].item() if "hallucination_flags" in data else 0.0),
                })

        logger.info(f"Loaded {len(self.samples)} total trajectories from {data_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class HybridCollateFn:
    """
    Collate function that dynamically tokenizes texts for a micro-batch
    and packages adjacency matrices and risk modifier tensors.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [item["tokenized_text"] for item in batch]
        adj_matrices = [item["adj_matrix"] for item in batch]
        risk_levels = torch.tensor([item["risk_level"] for item in batch], dtype=torch.float32)
        risk_modifiers = torch.tensor([item["risk_modifier"] for item in batch], dtype=torch.float32)
        sample_ids = [item["id"] for item in batch]

        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "adj_matrices": adj_matrices,
            "risk_levels": risk_levels,
            "risk_modifiers": risk_modifiers,
            "sample_ids": sample_ids,
        }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_accum_steps: int = 8,
    scaler: Optional[torch.amp.GradScaler] = None,
    epoch: int = 1,
) -> float:
    """
    Executes one training epoch with micro-batching and gradient accumulation.
    """
    model.train()
    total_loss = 0.0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    num_batches = len(dataloader)

    use_cuda_amp = (device.type == "cuda" and scaler is not None)

    for step, batch in enumerate(dataloader, start=1):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        adj_matrices = [adj.to(device) for adj in batch["adj_matrices"]]
        targets = batch["risk_levels"].to(device)
        risk_modifiers = batch["risk_modifiers"].to(device)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp):
            predictions = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                adj_matrices=adj_matrices,
                risk_modifiers=risk_modifiers,
            )
            raw_loss = criterion(predictions, targets)
            loss = raw_loss / grad_accum_steps

        if use_cuda_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += raw_loss.item()

        # Optimizer step upon reaching accumulation boundary
        if step % grad_accum_steps == 0 or step == num_batches:
            if use_cuda_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            total_loss += accumulated_loss
            accumulated_loss = 0.0

        if step % 20 == 0 or step == num_batches:
            logger.info(
                f"Epoch [{epoch}] Step [{step}/{num_batches}] - Current Loss: {raw_loss.item():.4f}"
            )

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluates model on validation set using Mean Absolute Error (MAE).
    """
    model.eval()
    total_mae = 0.0
    sample_records: List[Dict[str, Any]] = []

    use_cuda_amp = (device.type == "cuda")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            adj_matrices = [adj.to(device) for adj in batch["adj_matrices"]]
            targets = batch["risk_levels"].to(device)
            risk_modifiers = batch["risk_modifiers"].to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda_amp):
                predictions = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adj_matrices=adj_matrices,
                    risk_modifiers=risk_modifiers,
                )
                mae_loss = criterion(predictions, targets)

            total_mae += mae_loss.item() * len(targets)

            preds_np = predictions.detach().cpu().tolist()
            targs_np = targets.detach().cpu().tolist()
            mods_np = risk_modifiers.detach().cpu().tolist()
            ids = batch["sample_ids"]

            for i in range(len(ids)):
                sample_records.append({
                    "id": ids[i],
                    "predicted_risk": round(preds_np[i], 3),
                    "target_risk": round(targs_np[i], 3),
                    "risk_modifier": round(mods_np[i], 3),
                    "abs_error": round(abs(preds_np[i] - targs_np[i]), 3),
                })

    avg_mae = total_mae / max(len(dataloader.dataset), 1)
    return avg_mae, sample_records


def run_training():
    parser = argparse.ArgumentParser(
        description="Phase 3: Hybrid Llama 3 + GNN Model Training with LoRA and MAE Loss."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="Hugging Face model identifier or path for Llama 3.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_2_verified" / "train"),
        help="Path to Phase 2 verified train batches.",
    )
    parser.add_argument(
        "--dev_data_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_2_verified" / "dev"),
        help="Path to Phase 2 verified dev batches.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "models" / "lora_adapters"),
        help="Directory to save the best model adapters and checkpoints.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs (default: 5).",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=2,
        help="VRAM-optimized micro-batch size (default: 2).",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps for effective batch size 16 (default: 8).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA and fusion regression head (default: 2e-4).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="Maximum sequence length for tokenized interaction string.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Runs a rapid dry-run to verify training pass without full epochs.",
    )

    args = parser.parse_args()

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    logger.info(f"Target Compute Device: {device} (Available GPUs: {num_gpus})")
    if device.type == "cuda":
        for i in range(num_gpus):
            logger.info(f"  GPU [{i}]: {torch.cuda.get_device_name(i)} (VRAM: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB)")

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Initialize Tokenizer
    logger.info(f"Loading Tokenizer for {args.model_name_or_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    except Exception as e:
        logger.warning(f"Could not load remote tokenizer ({e}); falling back to standard LlamaTokenizerFast.")
        from transformers import LlamaTokenizerFast
        tokenizer = LlamaTokenizerFast.from_pretrained("hf-internal-testing/llama-tokenizer")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<|pad|>"

    # 2. Build Datasets and DataLoaders
    train_dataset = VerifiedTrajectoryDataset(Path(args.train_data_dir))
    dev_dataset = VerifiedTrajectoryDataset(Path(args.dev_data_dir))

    collate_fn = HybridCollateFn(tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.micro_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    # 3. Instantiate HybridRiskModel
    logger.info(f"Instantiating HybridRiskModel (LoRA r=8, alpha=16) on {device}...")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    model = HybridRiskModel(
        model_name_or_path=args.model_name_or_path,
        gnn_in_features=16,
        gnn_hidden_dim=64,
        gnn_out_features=128,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        torch_dtype=torch_dtype,
        load_pretrained=True,
    ).to(device)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Model Parameters: {total_params:,} total | {trainable_params:,} trainable "
        f"({100 * trainable_params / total_params:.2f}% active)"
    )

    # 4. Optimizer, Scaler, and Criterion (Mean Absolute Error)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    criterion = nn.L1Loss()  # Evaluated using Mean Absolute Error (MAE)

    best_val_mae = float("inf")
    training_manifest = {
        "task": "SemEval 2027 Task 10 (AgentRisk) - Phase 3 Training",
        "model_name_or_path": args.model_name_or_path,
        "lora_rank": 8,
        "lora_alpha": 16,
        "micro_batch_size": args.micro_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.micro_batch_size * args.grad_accum_steps,
        "device": str(device),
        "history": [],
    }

    total_epochs = 1 if args.dry_run else args.epochs
    logger.info(f"Starting Training for {total_epochs} epoch(s)...")

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            epoch=epoch,
        )

        val_mae, val_records = evaluate(
            model=model,
            dataloader=dev_loader,
            criterion=criterion,
            device=device,
        )

        epoch_time = time.time() - epoch_start
        logger.info(
            f"Epoch [{epoch}/{total_epochs}] Finished in {epoch_time:.1f}s | "
            f"Train MAE Loss: {train_loss:.4f} | Dev MAE: {val_mae:.4f}"
        )

        is_best = val_mae < best_val_mae
        if is_best:
            best_val_mae = val_mae
            logger.info(f"[*] New best Dev MAE ({best_val_mae:.4f})! Saving adapters to {output_dir}...")
            model.save_adapters(output_dir)

        epoch_summary = {
            "epoch": epoch,
            "train_mae": round(train_loss, 4),
            "dev_mae": round(val_mae, 4),
            "is_best": is_best,
            "epoch_duration_seconds": round(epoch_time, 2),
        }
        training_manifest["history"].append(epoch_summary)

        # Clear memory between epochs
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    training_manifest["best_val_mae"] = round(best_val_mae, 4)
    manifest_path = output_dir / "training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(training_manifest, f, indent=2)

    logger.info("=" * 65)
    logger.info("Phase 3 Training Complete!")
    logger.info(f"Best Validation MAE : {best_val_mae:.4f}")
    logger.info(f"Adapters Saved To   : {output_dir}")
    logger.info(f"Training Manifest   : {manifest_path.name}")
    logger.info("=" * 65)


if __name__ == "__main__":
    run_training()


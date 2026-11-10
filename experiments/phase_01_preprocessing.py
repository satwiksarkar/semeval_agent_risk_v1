#!/usr/bin/env python3
"""
phase_01_preprocessing.py
Phase 1 Execution Script: Converts raw SemEval 2027 Task 10 interaction trajectories
into Directed Acyclic Graphs (DAGs), extracts topological adjacency matrices and
structurally tokenized text, and saves batches as PyTorch tensors (.pt files).

Hardware-Optimized for Core i3 and 8GB RAM:
- Iterative streaming architecture avoids loading large raw datasets into memory.
- Accumulates exactly 1,000 graphs in memory per batch before persisting to disk.
- Invokes garbage collection post-batch to keep RAM usage strictly below 250MB.
- Supports both multi-split (train/dev) directories and single raw data files.
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional

import torch

# Ensure src is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_risk.dag_parser import TrajectoryDAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Phase01Preprocessor")


def get_memory_mb() -> float:
    """Returns the current process memory consumption in megabytes."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def stream_json_lines(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Streams records line-by-line from a JSONL file to prevent RAM spikes."""
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at line {line_num} in {file_path}: {e}")


def stream_json_array_or_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Handles streaming from a JSON file (either a JSON array or single object)."""
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    yield item
            elif isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        yield item
                elif "instances" in data and isinstance(data["instances"], list):
                    for item in data["instances"]:
                        yield item
                else:
                    yield data
        except Exception as e:
            logger.error(f"Error reading JSON from {file_path}: {e}")


def stream_file_records(file_path: Path) -> Generator[Dict[str, Any], None, None]:
    """Yields records from a single file based on extension."""
    if file_path.suffix.lower() == ".jsonl":
        yield from stream_json_lines(file_path)
    elif file_path.suffix.lower() == ".json":
        yield from stream_json_array_or_file(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def generate_synthetic_samples(num_samples: int = 2500) -> Generator[Dict[str, Any], None, None]:
    """
    Generates synthetic agent trajectories with realistic actions, observations,
    dead-end error branches, and connected errors for pipeline testing and verification.
    """
    tools = ["bash", "file_reader", "sql_query", "web_search", "git_cli", "curl"]
    logger.info(f"Generating {num_samples} synthetic agent trajectories for testing...")

    for i in range(1, num_samples + 1):
        steps = []
        has_error_branch = (i % 3 == 0)
        has_connected_error = (i % 5 == 0)

        steps.append({
            "step_id": 1,
            "thought": f"Gather context for task {i}",
            "action": {"tool": tools[i % len(tools)], "inputs": {"query": f"init_param_{i}"}},
            "observation": f"Successfully initialized step 1 for item {i}",
            "is_error": False,
        })

        if has_error_branch:
            steps.append({
                "step_id": 2,
                "thought": "Attempt alternative speculative branch",
                "action": {"tool": "curl", "inputs": {"url": "http://192.168.1.999/invalid"}},
                "observation": "Error: 404 Not Found - host unreachable",
                "is_error": True,
                "dependencies": [1],
            })

        if has_connected_error:
            steps.append({
                "step_id": 3,
                "thought": "Try reading protected file directly",
                "action": {"tool": "cat", "inputs": {"path": "/etc/shadow"}},
                "observation": "Error: Permission denied (errno 13)",
                "is_error": True,
                "dependencies": [1],
            })
            steps.append({
                "step_id": 4,
                "thought": "Permission denied; pivot to privilege escalation",
                "action": {"tool": "bash", "inputs": {"cmd": "sudo -u root bash -i"}},
                "observation": "Elevated session opened.",
                "is_error": False,
                "dependencies": [3],
            })
        else:
            steps.append({
                "step_id": 3,
                "thought": f"Finalize parameters for execution {i}",
                "action": {"tool": tools[(i + 1) % len(tools)], "inputs": {"filter": "active"}},
                "observation": f"Data retrieved successfully: record_count={i*2}",
                "is_error": False,
                "dependencies": [1],
            })

        risk_level = (i % 5) + 1
        sample = {
            "example_id": f"AGENTRISK-SYNTH-{i:06d}",
            "goal": f"Execute administrative procedure {i}",
            "trajectory": steps,
            "proposed_action": {
                "tool": "bash" if risk_level >= 4 else "file_writer",
                "inputs": {"cmd": f"run_eval_{i}.sh --force"} if risk_level >= 4 else {"dest": f"log_{i}.txt"},
                "description": f"Proposed execution for item {i}",
            },
            "risk_severity": risk_level,
        }
        yield sample


def save_batch(
    batch_records: List[Dict[str, Any]],
    batch_idx: int,
    output_dir: Path,
    prefix: str = "batch",
) -> Path:
    """
    Persists exactly one batch of processed DAGs as a PyTorch .pt tensor file.
    """
    batch_file = output_dir / f"{prefix}_{batch_idx:05d}.pt"

    batch_payload = {
        "batch_index": batch_idx,
        "num_samples": len(batch_records),
        "sample_ids": [r["id"] for r in batch_records],
        "tokenized_texts": [r["tokenized_text"] for r in batch_records],
        "adj_matrices": [torch.tensor(r["adjacency_matrix"], dtype=torch.float32) for r in batch_records],
        "risk_levels": torch.tensor([r["risk_level"] for r in batch_records], dtype=torch.float32),
        "num_nodes": torch.tensor([r["num_nodes"] for r in batch_records], dtype=torch.long),
        "num_edges": torch.tensor([r["num_edges"] for r in batch_records], dtype=torch.long),
        "risk_domains": [r.get("risk_domains", []) for r in batch_records],
        "risk_drivers": [r.get("risk_drivers", []) for r in batch_records],
    }

    torch.save(batch_payload, batch_file)
    logger.info(
        f"Saved Batch {batch_idx:05d} ({len(batch_records)} graphs) -> {batch_file.name} "
        f"[RAM: {get_memory_mb():.1f} MB]"
    )
    return batch_file


def process_stream(
    dataset_stream: Iterator[Dict[str, Any]],
    output_dir: Path,
    split_name: str = "dataset",
    batch_size: int = 1000,
) -> Dict[str, Any]:
    """
    Processes a stream of trajectories into DAGs iteratively, accumulating exactly
    batch_size graphs in memory before persisting to prevent RAM overflow on 8GB machines.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_records: List[Dict[str, Any]] = []
    batch_idx = 0
    total_processed = 0
    total_pruned_nodes = 0
    total_nodes = 0
    total_edges = 0
    risk_distribution: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    batch_files: List[str] = []

    start_time = time.time()
    logger.info(f"[{split_name.upper()}] Starting DAG Preprocessing (batch threshold: {batch_size} graphs)...")

    for record in dataset_stream:
        try:
            dag = TrajectoryDAG(record, scale_to_1_5=True)
            parsed_dict = dag.to_dict()

            batch_records.append(parsed_dict)
            total_processed += 1
            total_pruned_nodes += parsed_dict["pruned_nodes_count"]
            total_nodes += parsed_dict["num_nodes"]
            total_edges += parsed_dict["num_edges"]

            risk_int = int(round(parsed_dict["risk_level"]))
            risk_int = max(1, min(5, risk_int))
            risk_distribution[risk_int] = risk_distribution.get(risk_int, 0) + 1

            if len(batch_records) == batch_size:
                saved_path = save_batch(batch_records, batch_idx, output_dir, prefix="batch")
                batch_files.append(saved_path.name)
                batch_idx += 1

                batch_records.clear()
                del parsed_dict, dag
                gc.collect()

        except Exception as e:
            rec_id = record.get("example_id") or record.get("id", "unknown")
            logger.error(f"Error processing record {rec_id}: {e}", exc_info=True)

    # Persist any remaining graphs in final batch
    if batch_records:
        saved_path = save_batch(batch_records, batch_idx, output_dir, prefix="batch")
        batch_files.append(saved_path.name)
        batch_idx += 1
        batch_records.clear()
        gc.collect()

    elapsed = time.time() - start_time
    avg_nodes = (total_nodes / total_processed) if total_processed > 0 else 0.0
    avg_edges = (total_edges / total_processed) if total_processed > 0 else 0.0

    manifest = {
        "task": "SemEval 2027 Task 10 (AgentRisk) - Subtask 1",
        "split": split_name,
        "phase": "Phase 1 - DAG Preprocessing",
        "total_trajectories_processed": total_processed,
        "batch_size_threshold": batch_size,
        "total_batches_saved": batch_idx,
        "batch_files": batch_files,
        "average_nodes_per_graph": round(avg_nodes, 2),
        "average_edges_per_graph": round(avg_edges, 2),
        "total_dead_end_nodes_pruned": total_pruned_nodes,
        "risk_level_distribution": risk_distribution,
        "processing_time_seconds": round(elapsed, 2),
        "samples_per_second": round(total_processed / max(elapsed, 0.001), 1),
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"[{split_name.upper()}] Processed: {total_processed} | Batches: {batch_idx} | "
                f"Avg nodes: {avg_nodes:.2f} | Speed: {manifest['samples_per_second']} samples/s")

    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Preprocess SemEval 2027 AgentRisk Trajectories into DAG PyTorch Tensors."
    )
    parser.add_argument(
        "--raw_data_path",
        type=str,
        default=None,
        help="Path to raw dataset (.json, .jsonl, or directory). Defaults to data/raw/",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "processed" / "phase_1_dag"),
        help="Output directory for processed .pt batches. Defaults to data/processed/phase_1_dag/",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Exact number of graphs to accumulate in memory before writing a batch (default: 1000).",
    )
    parser.add_argument(
        "--generate_dummy",
        type=int,
        default=None,
        help="Generate N synthetic samples for dry-run verification (e.g. --generate_dummy 2500).",
    )

    args = parser.parse_args()
    base_output_dir = Path(args.output_dir)

    # 1. Synthetic dummy mode
    if args.generate_dummy is not None:
        logger.info(f"Running synthetic generation mode with {args.generate_dummy} samples.")
        stream = generate_synthetic_samples(args.generate_dummy)
        process_stream(stream, base_output_dir, split_name="synthetic", batch_size=args.batch_size)
        return

    # 2. Check input path
    raw_path = Path(args.raw_data_path) if args.raw_data_path else PROJECT_ROOT / "data" / "raw"

    if raw_path.is_file():
        # Single file execution
        split_name = raw_path.stem
        stream = stream_file_records(raw_path)
        process_stream(stream, base_output_dir, split_name=split_name, batch_size=args.batch_size)

    elif raw_path.is_dir():
        raw_files = sorted(raw_path.glob("*.json*"))
        raw_files = [f for f in raw_files if f.name != ".gitkeep"]

        if not raw_files:
            logger.warning(
                f"No raw files found in {raw_path}. Running synthetic verification (2,500 trajectories)."
            )
            stream = generate_synthetic_samples(2500)
            process_stream(stream, base_output_dir, split_name="synthetic", batch_size=args.batch_size)
            return

        # Multi-file / split processing: process each split into its designated subfolder
        # e.g., data/processed/phase_1_dag/train/ and data/processed/phase_1_dag/dev/
        overall_manifest = {
            "task": "SemEval 2027 Task 10 (AgentRisk) - Subtask 1",
            "phase": "Phase 1 - DAG Preprocessing Summary",
            "splits": {},
        }

        for file_path in raw_files:
            split_name = file_path.stem  # e.g. "train" or "dev"
            split_output_dir = base_output_dir / split_name
            stream = stream_file_records(file_path)
            split_manifest = process_stream(
                dataset_stream=stream,
                output_dir=split_output_dir,
                split_name=split_name,
                batch_size=args.batch_size,
            )
            overall_manifest["splits"][split_name] = split_manifest

        # Also write overall manifest in base output directory
        base_manifest_path = base_output_dir / "dataset_manifest.json"
        with open(base_manifest_path, "w", encoding="utf-8") as f:
            json.dump(overall_manifest, f, indent=2)

        logger.info("=" * 60)
        logger.info("Phase 1 Preprocessing Complete for all splits!")
        for s_name, s_info in overall_manifest["splits"].items():
            logger.info(
                f"  Split '{s_name}': {s_info['total_trajectories_processed']} trajectories -> "
                f"{s_info['total_batches_saved']} batch file(s) in {base_output_dir / s_name}"
            )
        logger.info(f"Overall manifest saved to: {base_manifest_path.name}")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()

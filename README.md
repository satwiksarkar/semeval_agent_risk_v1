# SemEval 2027 Task 10 (AgentRisk) - Subtask 1

This repository implements a **4-Phase Hybrid Architecture** for predicting the risk level of autonomous agent actions from multi-turn interaction trajectories on a 1–5 ordinal scale evaluated using Mean Absolute Error (MAE).

> 📖 **Comprehensive Walkthrough:** For complete mathematical formulation, system diagrams, and phase-by-phase design rationales, refer to [METHODOLOGY.md](METHODOLOGY.md).

---

## 4-Phase System Architecture

```
                                  [Raw Trajectory JSON]
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Phase 1: TrajectoryDAG Parser (NetworkX)       │
                    │ - Directed Causal Modeling (API -> Obs)       │
                    │ - Dead-End Error Branch Pruning               │
                    │ - Topological Adjacency Matrix & Special Tags │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Phase 2: LangGraph Action Verifier            │
                    │ - Dangling Call Heuristics (+0.5 risk)        │
                    │ - Ignored Error Verification (+1.0 risk)      │
                    │ - Enriches batches with hallucination tensors │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Phase 3: Parallel Hybrid PyTorch Net          │
                    │ - Llama 3 (LoRA r=8, alpha=16) for Text       │
                    │ - VanillaGCN (2-layer graph conv) for DAG     │
                    │ - Multimodal Fusion Regression Head (MAE Loss)│
                    │ - Physical Risk Modifier Penalty Addition     │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │ Phase 4: Constrained Decoding                 │
                    │ - Outlines CFG / Regex Enforcer               │
                    │ - Strict 1–5 Integer Output                   │
                    └───────────────────────────────────────────────┘
```

---

## Repository Structure

```
semeval_agent_risk_v1/
├── pyproject.toml                     # PEP 621 dependencies (no XGBoost / scikit-learn)
├── README.md                          # Full system documentation & execution commands
├── METHODOLOGY.md                     # In-depth architectural & mathematical walkthrough with flowcharts
├── .gitignore                         # Excludes checkpoints, raw data, and tensor batches
├── setup_structure.py                 # Cross-platform Python setup automation script
├── setup_structure.sh                 # POSIX Bash setup automation script
├── data/
│   ├── raw/                           # Raw SemEval 2027 dataset files (train.jsonl, dev.jsonl)
│   └── processed/
│       ├── phase_1_dag/               # Phase 1 DAG batches (train/ & dev/)
│       └── phase_2_verified/          # Phase 2 LangGraph verified batches (train/ & dev/)
├── models/
│   └── lora_adapters/                 # Checkpoints of best LoRA adapters & fusion heads
├── src/
│   └── agent_risk/
│       ├── __init__.py                # Package-level exports
│       ├── dag_parser.py              # TrajectoryDAG: NetworkX causal parser & pruner
│       ├── simulator/                 # Phase 2: LangGraph deterministic state verifier
│       │   ├── __init__.py
│       │   ├── state.py               # SimulationState TypedDict
│       │   └── verifier_nodes.py      # verify_trajectory: heuristic rules & risk modifiers
│       └── models/                    # Phase 3 & 4: PyTorch architectures & constrained decoding
│           ├── __init__.py
│           ├── gnn_encoder.py         # VanillaGCN: 2-layer graph conv without PyG
│           ├── hybrid_llama.py        # HybridRiskModel: Llama 3 + LoRA + GCN + Regression Head
│           └── constrained_decoder.py # ConstrainedRiskDecoder: Outlines [1-5] integer enforcer
├── experiments/
│   ├── __init__.py
│   ├── phase_01_preprocessing.py      # Phase 1 DAG batch processor (1,000 graphs/batch)
│   ├── phase_02_langgraph.py          # Phase 2 LangGraph batch verifier & feature enricher
│   ├── phase_03_hybrid_gnn.py         # Phase 3 Kaggle-optimized training loop
│   └── phase_04_inference.py          # Phase 4 Outlines inference, evaluation & submission
└── tests/
    ├── __init__.py
    ├── test_dag_parser.py             # Validation suite for DAG parsing and pruning logic
    ├── test_hybrid_model.py           # Unit tests for VanillaGCN & HybridRiskModel
    └── test_phase4_decoder.py         # Tests for adapter reload & Outlines constrained decoding
```

---

## Benchmark & Evaluation Results

The system was evaluated on the official **SemEval 2027 Task 10 Subtask 1 Dev Set** (143 multi-turn trajectories). Predictions were strictly constrained to the $\{1, 2, 3, 4, 5\}$ ordinal range via grammar-guided decoding.

### Primary Metrics Summary

| Metric | Score | Scale / Type | Notes |
|:---|:---:|:---:|:---|
| **Strict Integer MAE** | **0.3427** | 1–5 discrete | **Official Competition Metric** |
| Continuous Hybrid MAE | **0.3966** | Continuous | Pre-rounding regression score |
| **Exact Match Accuracy** | **68.53%** | Percentage | 98 / 143 exact risk matches |
| **Within-1 ($\pm 1$) Accuracy** | **97.20%** | Percentage | 139 / 143 within $\le 1$ risk point |

### Baseline Comparison

| Strategy / Model | Dev MAE | Error Reduction vs Baseline |
|:---|:---:|:---:|
| Random Guessing (Uniform $y \in \{1, 2, 3, 4, 5\}$) | $\approx 1.6000$ | — |
| Always Predict Mode ($y = 1$) | $1.2517$ | — |
| Always Predict Mean ($y \approx 2.08$) | $1.0924$ | — |
| **4-Phase Hybrid Model (Ours)** | **0.3427** | **68.6% error reduction over mean baseline** |

### Per-Class Risk Breakdown

| Risk Level | Sample Count | Integer MAE | Exact Match Accuracy |
|:---:|:---:|:---:|:---:|
| **Class 1 (Negligible Risk)** | 66 | **0.0303** | **96.97%** |
| **Class 2 (Low Risk)** | 8 | 0.7500 | 37.50% |
| **Class 3 (Moderate Risk)** | 32 | 0.9062 | 18.75% |
| **Class 4 (High Risk)** | 35 | 0.3429 | 65.71% |
| **Class 5 (Critical Risk)** | 2 | **0.0000** | **100.00%** |

### Confusion Matrix ($5 \times 5$)

```
                Predicted Risk Level
             1      2      3      4      5
Actual  1 [ 64      2      0      0      0 ]  (96.97% correct)
Class   2 [  4      3      0      1      0 ]  (37.50% correct)
        3 [  0      1      6     22      3 ]  (18.75% correct; 68.8% conservatively classified as 4)
        4 [  0      0      7     23      5 ]  (65.71% correct)
        5 [  0      0      0      0      2 ]  (100.0% correct)
```

### Kaggle Training History (5 Epochs on 2x T4 GPU)

| Epoch | Train MAE Loss | Dev MAE | Best Checkpoint | Time per Epoch |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 0.7231 | 0.4591 | Saved | ~1997s |
| **2** | 0.4573 | 0.5063 | — | ~2012s |
| **3** | 0.4108 | 0.4136 | Saved | ~2013s |
| **4** | **0.3142** | **0.3965** | **Best Model Checkpoint** | ~2027s |
| **5** | 0.3002 | 0.5770 | — | ~2018s |

---

## Execution Guide

### 1. Phase 1: DAG Preprocessing
```powershell
python semeval_agent_risk_v1\experiments\phase_01_preprocessing.py
```
Converts raw interaction traces into topological causal DAGs and saves batch tensors to `data/processed/phase_1_dag/`.

### 2. Phase 2: LangGraph Action Verifier
```powershell
python semeval_agent_risk_v1\experiments\phase_02_langgraph.py
```
Applies deterministic hallucination checks (dangling calls and ignored errors) and appends `hallucination_flags` and `risk_modifiers` to `data/processed/phase_2_verified/`.

### 3. Phase 3: Hybrid Llama 3 + GNN Training (Kaggle 2x T4 GPU)
```bash
python experiments/phase_03_hybrid_gnn.py \
    --model_name_or_path "meta-llama/Meta-Llama-3-8B" \
    --train_data_dir "data/processed/phase_2_verified/train" \
    --dev_data_dir "data/processed/phase_2_verified/dev" \
    --output_dir "models/lora_adapters" \
    --device_map "auto" \
    --load_in_4bit \
    --epochs 5 \
    --micro_batch_size 2 \
    --grad_accum_steps 8 \
    --max_seq_len 1024 \
    --lr 2e-4
```
- **VRAM Optimization**: Uses mixed precision (`torch.autocast` in fp16), micro-batch size of 2, and 8 gradient accumulation steps (effective batch size: 16).
- **Physical Penalty**: `final_pred = regression_head(cat(h_text, h_graph)) + risk_modifier`.
- **Checkpointing**: Evaluates using Mean Absolute Error (`nn.L1Loss()`) and saves the best model adapters to `models/lora_adapters/`.

### 4. Phase 4: Constrained Decoding (`outlines`) & Evaluation
- **Evaluate on `dev` (computes Continuous MAE, Strict `1–5` Integer MAE, Exact Accuracy %, Within-1 %, and 5x5 Confusion Matrix):**
  ```bash
  python experiments/phase_04_inference.py \
      --mode eval \
      --adapter_dir "models/lora_adapters" \
      --data_path "data/processed/phase_2_verified/dev" \
      --output_dir "outputs" \
      --device_map "auto" \
      --load_in_4bit
  ```

- **End-to-End 4-Phase Prediction on Raw Test JSONL (generates `submission.csv` & `predictions.jsonl`):**
  ```bash
  python experiments/phase_04_inference.py \
      --mode predict \
      --adapter_dir "models/lora_adapters" \
      --data_path "data/raw/test.jsonl" \
      --output_dir "outputs" \
      --device_map "auto" \
      --load_in_4bit
  ```

---

## License

This project is licensed under the terms of the MIT License as specified in [pyproject.toml](file:///c:/Semeval_developments/semeval_agent_risk_v1/pyproject.toml).


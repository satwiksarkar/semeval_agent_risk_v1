"""
test_hybrid_model.py
Unit tests for VanillaGCN, HybridRiskModel architecture, last-token pooling, and left-truncation.
"""

import sys
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_risk.models.gnn_encoder import VanillaGCN
from agent_risk.models.hybrid_llama import HybridRiskModel
from experiments.phase_03_hybrid_gnn import HybridCollateFn
from transformers import LlamaConfig


def test_vanilla_gcn():
    gcn = VanillaGCN(in_features=16, hidden_dim=32, out_features=64)

    # 1. Single graph: 5 nodes
    adj_single = torch.tensor([
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0],
    ], dtype=torch.float32)

    emb_single = gcn(adj_single)
    assert emb_single.shape == (1, 64)

    # 2. Batch of graphs with different node counts
    adj_3 = torch.tensor([
        [0, 1, 1],
        [0, 0, 1],
        [0, 0, 0],
    ], dtype=torch.float32)

    batch_adj = [adj_single, adj_3]
    emb_batch = gcn(batch_adj)
    assert emb_batch.shape == (2, 64)
    print("[PASS] VanillaGCN tests passed successfully!")


def test_hybrid_risk_model_with_dummy_backbone():
    tiny_config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=256,
    )

    model = HybridRiskModel(
        model_name_or_path="mock-llama",
        config=tiny_config,
        gnn_in_features=16,
        gnn_hidden_dim=32,
        gnn_out_features=64,
        lora_r=4,
        lora_alpha=8,
    )

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(1, 1000, (batch_size, seq_len))
    # Keep identical token 0 across both samples so we verify pooling depends on later tokens
    input_ids[:, 0] = 1
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    adj_list = [
        torch.eye(4),
        torch.eye(6),
    ]
    risk_modifiers = torch.tensor([0.5, 1.0], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask, adj_list, risk_modifiers)
        assert output.shape == (batch_size,)

        # 1. Verify risk modifier physical penalty
        output_no_mod = model(input_ids, attention_mask, adj_list, torch.zeros(batch_size))
        diff = (output - output_no_mod).detach()
        assert torch.allclose(diff, risk_modifiers, atol=1e-4)

        # 2. Verify causal last-token pooling responds to downstream tokens (not stuck at token 0)
        altered_ids = input_ids.clone()
        altered_ids[:, -1] = (altered_ids[:, -1] + 50) % 999 + 1
        output_altered = model(altered_ids, attention_mask, adj_list, torch.zeros(batch_size))
        assert not torch.allclose(output_no_mod, output_altered, atol=1e-6), (
            "Model output did not change when downstream tokens changed!"
        )

        # 3. Verify right-padding invariance on last non-padding token pooling
        unpadded_ids = input_ids[:1, :8]
        unpadded_mask = torch.ones((1, 8), dtype=torch.long)
        padded_ids = torch.cat([unpadded_ids, torch.zeros((1, 8), dtype=torch.long)], dim=1)
        padded_mask = torch.cat([unpadded_mask, torch.zeros((1, 8), dtype=torch.long)], dim=1)
        out_unpadded = model(unpadded_ids, unpadded_mask, [adj_list[0]], torch.zeros(1))
        out_padded = model(padded_ids, padded_mask, [adj_list[0]], torch.zeros(1))
        assert torch.allclose(out_unpadded, out_padded, atol=1e-5)

    print("[PASS] HybridRiskModel pooling, padding invariance, and risk modifier verified!")


def test_left_truncation_in_collate():
    class DummyTokenizer:
        def __init__(self):
            self.truncation_side = "right"

        def __call__(self, texts, padding=True, truncation=True, max_length=1024, return_tensors="pt"):
            assert self.truncation_side == "left"
            tokens_list = [t.split() for t in texts]
            if truncation:
                if self.truncation_side == "left":
                    tokens_list = [toks[-max_length:] for toks in tokens_list]
                else:
                    tokens_list = [toks[:max_length] for toks in tokens_list]
            # Encode words to simple deterministic ints for test
            vocab = {}
            ids_batch = []
            max_l = max(len(t) for t in tokens_list)
            masks_batch = []
            for toks in tokens_list:
                row = [vocab.setdefault(w, len(vocab) + 1) for w in toks]
                pad = max_l - len(row)
                ids_batch.append(row + [0] * pad)
                masks_batch.append([1] * len(row) + [0] * pad)
            return {
                "input_ids": torch.tensor(ids_batch, dtype=torch.long),
                "attention_mask": torch.tensor(masks_batch, dtype=torch.long),
            }

    tok = DummyTokenizer()
    collate = HybridCollateFn(tokenizer=tok, max_seq_len=4)
    assert tok.truncation_side == "left"

    batch = [{
        "id": "ex-1",
        "tokenized_text": "step1 step2 step3 step4 <|proposed_action|> rm -rf",
        "adj_matrix": torch.eye(3),
        "risk_level": 5.0,
        "risk_modifier": 0.0,
    }]
    out = collate(batch)
    assert out["input_ids"].shape == (1, 4)
    print("[PASS] HybridCollateFn left-truncation verified!")


if __name__ == "__main__":
    test_vanilla_gcn()
    test_hybrid_risk_model_with_dummy_backbone()
    test_left_truncation_in_collate()
    print("\nAll Phase 3 architecture tests passed!")

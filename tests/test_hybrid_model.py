"""
test_hybrid_model.py
Unit tests for VanillaGCN and HybridRiskModel architecture.
"""

import sys
from pathlib import Path
import torch

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent_risk.models.gnn_encoder import VanillaGCN
from agent_risk.models.hybrid_llama import HybridRiskModel
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
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len))
    adj_list = [
        torch.eye(4),
        torch.eye(6),
    ]
    risk_modifiers = torch.tensor([0.5, 1.0], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(input_ids, attention_mask, adj_list, risk_modifiers)
        assert output.shape == (batch_size,)

        # Verify risk modifier physical penalty
        output_no_mod = model(input_ids, attention_mask, adj_list, torch.zeros(batch_size))
        diff = (output - output_no_mod).detach()
        assert torch.allclose(diff, risk_modifiers, atol=1e-4)
    print("[PASS] HybridRiskModel forward pass and risk modifier addition verified!")


if __name__ == "__main__":
    test_vanilla_gcn()
    test_hybrid_risk_model_with_dummy_backbone()
    print("\nAll Phase 3 architecture tests passed!")

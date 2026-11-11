"""
test_phase4_decoder.py
Unit and integration tests for Phase 4 Constrained Decoding, Adapter Round-Trip,
shared CausalLM backbone inference, and Evaluation Metrics.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent_risk.models.constrained_decoder import ConstrainedRiskDecoder
from agent_risk.models.hybrid_llama import HybridRiskModel
from experiments.phase_04_inference import (
    compute_evaluation_metrics,
    preprocess_raw_to_phase2_records,
)


class SimpleDigitTokenizer:
    """Minimal mock tokenizer supporting digits '1'..'5' for offline testing."""

    def __init__(self):
        self.vocab = {"<pad>": 0, "<bos>": 1, "1": 11, "2": 12, "3": 13, "4": 14, "5": 15}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = 0
        self.truncation_side = "left"

    def encode(self, text, add_special_tokens=False):
        if text in self.vocab:
            return [self.vocab[text]]
        return [1 + (abs(hash(w)) % 10) for w in text.split()][:32]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.inv_vocab.get(i, "?") for i in ids)

    def __call__(self, text, return_tensors="pt", truncation=True, max_length=768):
        ids = self.encode(text)[-max_length:]
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


def test_adapter_save_and_reload_roundtrip():
    tiny_config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=128,
    )

    base_orig = LlamaForCausalLM(tiny_config)
    causal_lm_reload = LlamaForCausalLM(tiny_config)
    causal_lm_reload.load_state_dict(base_orig.state_dict())

    model = HybridRiskModel(
        model_name_or_path="mock-llama",
        base_model=base_orig.model,
        gnn_in_features=16,
        gnn_hidden_dim=16,
        gnn_out_features=32,
        lora_r=4,
        lora_alpha=8,
        torch_dtype=torch.float32,
    )
    # Perturb LoRA weights so reload test proves non-zero LoRA delta is loaded accurately
    with torch.no_grad():
        for name, param in model.llm.named_parameters():
            if "lora_" in name:
                param.add_(0.05)
    model.eval()

    input_ids = torch.randint(1, 20, (2, 10))
    attention_mask = torch.ones((2, 10), dtype=torch.long)
    adj_list = [torch.eye(4), torch.eye(5)]
    risk_mods = torch.tensor([0.0, 0.5], dtype=torch.float32)

    with torch.no_grad():
        orig_out = model(input_ids, attention_mask, adj_list, risk_mods)

    tmp_dir = Path(tempfile.mkdtemp(prefix="agent_risk_ckpt_"))
    try:
        model.save_adapters(tmp_dir)

        reloaded = HybridRiskModel.from_adapters(
            adapter_dir=tmp_dir,
            base_model=causal_lm_reload.model,
            torch_dtype=torch.float32,
        )
        reloaded.eval()

        with torch.no_grad():
            reloaded_out = reloaded(input_ids, attention_mask, adj_list, risk_mods)

        assert torch.allclose(orig_out, reloaded_out, atol=1e-5), (
            f"Reloaded output {reloaded_out} differs from original {orig_out}"
        )

        # Now verify CausalLM.generate + ConstrainedRiskDecoder works with shared backbone
        tok = SimpleDigitTokenizer()
        decoder = ConstrainedRiskDecoder(
            causal_lm=causal_lm_reload,
            tokenizer=tok,
            hybrid_weight=1.0,
            lm_weight=0.05,
        )
        pred_int, diag = decoder.decode_single(
            hybrid_score=3.8,
            tokenized_text="<|user_intent|> test <|proposed_action|> rm -rf",
            hallucination_flag=1.0,
            risk_modifier=0.5,
        )
        assert isinstance(pred_int, int) and 1 <= pred_int <= 5
        assert pred_int == 4, f"Expected 4 for hybrid_score=3.8, got {pred_int}"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[PASS] Adapter save/reload round-trip & shared CausalLM constrained decoding passed!")


def test_strict_int_bounds_and_metrics():
    assert ConstrainedRiskDecoder.enforce_strict_int(-3.5) == 1
    assert ConstrainedRiskDecoder.enforce_strict_int(0.2) == 1
    assert ConstrainedRiskDecoder.enforce_strict_int(1.49) == 1
    assert ConstrainedRiskDecoder.enforce_strict_int(2.6) == 3
    assert ConstrainedRiskDecoder.enforce_strict_int(9.9) == 5
    assert ConstrainedRiskDecoder.enforce_strict_int("Score: 4") == 4
    assert ConstrainedRiskDecoder.enforce_strict_int(float("nan")) == 3

    sample_records = [
        {"continuous_score": 1.1, "predicted_risk": 1, "true_risk": 1.0},
        {"continuous_score": 3.8, "predicted_risk": 4, "true_risk": 4.0},
        {"continuous_score": 2.9, "predicted_risk": 3, "true_risk": 4.0},
    ]
    metrics = compute_evaluation_metrics(sample_records)
    assert metrics["num_samples"] == 3
    assert metrics["exact_match_accuracy_pct"] == 66.67
    assert metrics["within_1_accuracy_pct"] == 100.0
    print("[PASS] Strict 1-5 bounds & evaluation metrics verified!")


def test_raw_dev_end_to_end_preprocess():
    dev_jsonl = ROOT_DIR / "data" / "raw" / "dev.jsonl"
    if dev_jsonl.exists():
        records, has_labels = preprocess_raw_to_phase2_records(dev_jsonl)
        assert len(records) == 143
        assert has_labels is True
        print("[PASS] End-to-end Phases 1+2 preprocessing on raw dev.jsonl (143 trajectories) passed!")


if __name__ == "__main__":
    test_adapter_save_and_reload_roundtrip()
    test_strict_int_bounds_and_metrics()
    test_raw_dev_end_to_end_preprocess()
    print("\nAll Phase 4 constrained decoding tests passed!")

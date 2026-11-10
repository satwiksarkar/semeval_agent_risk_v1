"""
hybrid_llama.py
HybridRiskModel: Fuses LoRA-adapted Llama 3 language representations with
topological GCN DAG structural embeddings for multi-turn agent risk estimation.

SemEval 2027 Task 10 (AgentRisk) - Phase 3
Applies PEFT/LoRA (rank 8, alpha 16) to freeze base LLM weights for Kaggle 2x T4 GPU constraints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PretrainedConfig
from peft import LoraConfig, get_peft_model, TaskType

from agent_risk.models.gnn_encoder import VanillaGCN


class HybridRiskModel(nn.Module):
    """
    Parallel hybrid architecture:
    1. Llama 3 processes structurally tokenized text sequence -> CLS embedding.
    2. VanillaGCN processes the topological adjacency matrix -> Graph embedding.
    3. Fused representation [h_text; h_graph] passes through a regression head.
    4. Phase 2 deterministic risk_modifier is physically added to penalize tool hallucinations.
    """

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Meta-Llama-3-8B",
        base_model: Optional[nn.Module] = None,
        config: Optional[PretrainedConfig] = None,
        gnn_in_features: int = 16,
        gnn_hidden_dim: int = 64,
        gnn_out_features: int = 128,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        torch_dtype: torch.dtype = torch.float16,
        load_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.gnn_out_features = gnn_out_features

        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]

        # 1. Base LLM Backbone
        if base_model is not None:
            raw_llm = base_model
            llm_hidden_size = getattr(raw_llm.config, "hidden_size", 4096)
        elif config is not None:
            raw_llm = AutoModel.from_config(config)
            llm_hidden_size = config.hidden_size
        elif load_pretrained:
            raw_llm = AutoModel.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            llm_hidden_size = raw_llm.config.hidden_size
        else:
            conf = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            raw_llm = AutoModel.from_config(conf)
            llm_hidden_size = conf.hidden_size

        self.llm_hidden_size = llm_hidden_size

        # 2. Apply PEFT LoRA (rank 8, alpha 16) to freeze base weights and minimize VRAM
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.llm = get_peft_model(raw_llm, lora_config)

        # 3. Vanilla GCN for Graph Topological Encoding
        self.gcn = VanillaGCN(
            in_features=gnn_in_features,
            hidden_dim=gnn_hidden_dim,
            out_features=gnn_out_features,
        )

        # 4. Multimodal Fusion Regression Head
        fusion_dim = self.llm_hidden_size + self.gnn_out_features
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrices: Union[torch.Tensor, List[torch.Tensor]],
        risk_modifiers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass fusing Llama 3 text features and GCN DAG features.
        
        Args:
            input_ids: (B, SeqLen) token IDs.
            attention_mask: (B, SeqLen) attention mask.
            adj_matrices: List of (N_i, N_i) adjacency tensors or (B, N, N) tensor.
            risk_modifiers: (B,) Phase 2 deterministic risk penalties.
            
        Returns:
            Scalar predictions (B,) on the 1-5 ordinal scale.
        """
        # A. Llama 3 forward pass: extract CLS token embedding
        llm_outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        # Using index 0 as the sequence start / CLS representation
        h_text = llm_outputs.last_hidden_state[:, 0, :]  # Shape: (B, llm_hidden_size)

        # B. GCN forward pass: extract graph structural embedding
        h_graph = self.gcn(adj_matrices)  # Shape: (B, gnn_out_features)

        # Ensure matching device and dtype for fusion
        h_graph = h_graph.to(device=h_text.device, dtype=h_text.dtype)

        # C. Fusion: concatenate text and graph embeddings
        fused_features = torch.cat([h_text, h_graph], dim=-1)  # Shape: (B, fusion_dim)

        # D. Regression head: output base continuous risk score
        base_risk_score = self.regression_head(fused_features).squeeze(-1)  # Shape: (B,)

        # E. Physical penalty: add deterministic Phase 2 risk modifier directly
        if risk_modifiers is not None:
            modifiers = risk_modifiers.to(device=base_risk_score.device, dtype=base_risk_score.dtype).view(-1)
            final_risk_score = base_risk_score + modifiers
        else:
            final_risk_score = base_risk_score

        return final_risk_score

    def save_adapters(self, save_directory: Union[str, Path]) -> None:
        """
        Saves the PEFT LoRA adapter weights and the non-LLM trainable heads
        (GCN and regression head) to save_directory.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter weights
        self.llm.save_pretrained(save_dir)

        # Save GCN and regression head state dict
        fusion_state = {
            "gcn": self.gcn.state_dict(),
            "regression_head": self.regression_head.state_dict(),
            "llm_hidden_size": self.llm_hidden_size,
            "gnn_out_features": self.gnn_out_features,
            "model_name_or_path": self.model_name_or_path,
        }
        torch.save(fusion_state, save_dir / "fusion_heads.pt")

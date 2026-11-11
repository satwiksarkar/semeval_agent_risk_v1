"""
hybrid_llama.py
HybridRiskModel: Fuses LoRA-adapted Llama 3 language representations with
topological GCN DAG structural embeddings for multi-turn agent risk estimation.

SemEval 2027 Task 10 (AgentRisk) - Phase 3
Applies PEFT/LoRA (rank 8, alpha 16) to freeze base LLM weights for Kaggle 2x T4 GPU constraints.
Supports multi-GPU sharding (device_map="auto"), 4-bit QLoRA, and gradient checkpointing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PretrainedConfig
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training,
)

from agent_risk.models.gnn_encoder import VanillaGCN


class HybridRiskModel(nn.Module):
    """
    Parallel hybrid architecture:
    1. Llama 3 processes structurally tokenized text sequence -> last non-padding token embedding.
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
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = False,
        hf_token: Optional[str] = None,
        adapter_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.gnn_out_features = gnn_out_features
        self.is_sharded = device_map is not None

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
            load_kwargs: Dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True,
            }
            if device_map is not None:
                load_kwargs["device_map"] = device_map
            if hf_token is not None:
                load_kwargs["token"] = hf_token

            if load_in_4bit:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            raw_llm = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
            llm_hidden_size = raw_llm.config.hidden_size

            if load_in_4bit and adapter_dir is None:
                raw_llm = prepare_model_for_kbit_training(
                    raw_llm, use_gradient_checkpointing=gradient_checkpointing
                )
        else:
            conf = AutoConfig.from_pretrained(
                model_name_or_path, trust_remote_code=True, token=hf_token
            )
            raw_llm = AutoModel.from_config(conf)
            llm_hidden_size = conf.hidden_size

        if gradient_checkpointing and not load_in_4bit and adapter_dir is None:
            if hasattr(raw_llm, "gradient_checkpointing_enable"):
                raw_llm.gradient_checkpointing_enable()
            if hasattr(raw_llm, "enable_input_require_grads"):
                raw_llm.enable_input_require_grads()

        self.llm_hidden_size = llm_hidden_size

        # 2. Apply or Load PEFT LoRA (rank 8, alpha 16)
        if adapter_dir is not None:
            self.llm = PeftModel.from_pretrained(raw_llm, str(adapter_dir), is_trainable=False)
        else:
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.llm = get_peft_model(raw_llm, lora_config)

            # Keep LoRA trainable weights in float32 for AMP GradScaler stability
            for param in self.llm.parameters():
                if param.requires_grad and param.dtype != torch.float32:
                    param.data = param.data.float()

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

        # Load trained GCN & regression head weights if loading from adapters
        if adapter_dir is not None:
            fusion_path = Path(adapter_dir) / "fusion_heads.pt"
            if fusion_path.exists():
                fusion_state = torch.load(fusion_path, map_location="cpu", weights_only=False)
                self.gcn.load_state_dict(fusion_state["gcn"])
                self.regression_head.load_state_dict(fusion_state["regression_head"])

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
        # Align input_ids with the LLM's input embedding device (essential when device_map="auto")
        llm_input_device = self.llm.get_input_embeddings().weight.device
        input_ids = input_ids.to(llm_input_device)
        attention_mask = attention_mask.to(llm_input_device)

        # A. Llama 3 forward pass: extract last non-padding token embedding
        # Note: Causal LMs (Llama 3) only attend leftward, so index 0 cannot see
        # downstream trajectory tokens. Pooling the last valid token encodes the
        # full sequence through <|proposed_action|> regardless of padding side.
        llm_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden_states = llm_outputs.last_hidden_state  # Shape: (B, SeqLen, llm_hidden_size)
        batch_size, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=attention_mask.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        last_token_indices = (positions * (attention_mask > 0).long()).max(dim=1).values
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        h_text = hidden_states[
            batch_indices, last_token_indices.to(hidden_states.device)
        ]  # Shape: (B, llm_hidden_size)

        # B. GCN forward pass: extract graph structural embedding
        gcn_device = next(self.gcn.parameters()).device
        if isinstance(adj_matrices, list):
            adj_on_dev = [adj.to(gcn_device) for adj in adj_matrices]
        else:
            adj_on_dev = adj_matrices.to(gcn_device)
        h_graph = self.gcn(adj_on_dev)  # Shape: (B, gnn_out_features)

        # Move both embeddings to the regression head's device and dtype
        head_param = next(self.regression_head.parameters())
        head_device = head_param.device
        h_text = h_text.to(device=head_device, dtype=torch.float32)
        h_graph = h_graph.to(device=head_device, dtype=torch.float32)

        # C. Fusion: concatenate text and graph embeddings
        fused_features = torch.cat([h_text, h_graph], dim=-1)  # Shape: (B, fusion_dim)

        # D. Regression head: output base continuous risk score
        base_risk_score = self.regression_head(fused_features).squeeze(-1)  # Shape: (B,)

        # E. Physical penalty: add deterministic Phase 2 risk modifier directly
        if risk_modifiers is not None:
            modifiers = risk_modifiers.to(device=head_device, dtype=base_risk_score.dtype).view(-1)
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
            "gnn_in_features": self.gcn.in_features,
            "gnn_hidden_dim": self.gcn.hidden_dim,
            "gnn_out_features": self.gnn_out_features,
            "model_name_or_path": self.model_name_or_path,
        }
        torch.save(fusion_state, save_dir / "fusion_heads.pt")

    @classmethod
    def from_adapters(
        cls,
        adapter_dir: Union[str, Path],
        base_model: Optional[nn.Module] = None,
        config: Optional[PretrainedConfig] = None,
        torch_dtype: torch.dtype = torch.float16,
        device_map: Optional[Union[str, Dict[str, Any]]] = None,
        load_in_4bit: bool = False,
        hf_token: Optional[str] = None,
    ) -> "HybridRiskModel":
        """
        Reconstructs a trained HybridRiskModel from saved LoRA adapters and fusion_heads.pt.
        """
        adapter_path = Path(adapter_dir)
        fusion_path = adapter_path / "fusion_heads.pt"
        if not fusion_path.exists():
            raise FileNotFoundError(f"Missing fusion_heads.pt in {adapter_path}")

        fusion_state = torch.load(fusion_path, map_location="cpu", weights_only=False)
        model_name = fusion_state.get("model_name_or_path", "meta-llama/Meta-Llama-3-8B")
        gnn_out = int(fusion_state.get("gnn_out_features", 128))
        gc1_weight = fusion_state["gcn"]["gc1.linear.weight"]
        gnn_hidden = int(fusion_state.get("gnn_hidden_dim", gc1_weight.shape[0]))
        gnn_in = int(fusion_state.get("gnn_in_features", gc1_weight.shape[1]))

        instance = cls(
            model_name_or_path=model_name,
            base_model=base_model,
            config=config,
            gnn_in_features=gnn_in,
            gnn_hidden_dim=gnn_hidden,
            gnn_out_features=gnn_out,
            torch_dtype=torch_dtype,
            load_pretrained=(base_model is None and config is None),
            device_map=device_map,
            load_in_4bit=load_in_4bit,
            gradient_checkpointing=False,
            hf_token=hf_token,
            adapter_dir=adapter_path,
        )
        instance.eval()
        return instance

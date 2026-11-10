"""
gnn_encoder.py
Vanilla PyTorch Graph Convolutional Network (GCN) for structural trajectory DAG embedding.

SemEval 2027 Task 10 (AgentRisk) - Phase 3
Does not require torch-geometric, preventing environment and dependency conflicts.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
    """
    Computes symmetric normalized adjacency with self-loops:
    A_tilde = A + I
    D_tilde_inv_sqrt = D^(-1/2)
    Normalized = D^(-1/2) * A_tilde * D^(-1/2)
    """
    num_nodes = adj.size(0)
    identity = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
    a_tilde = adj + identity

    deg = torch.sum(a_tilde, dim=-1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-7), -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    deg_mat = torch.diag(deg_inv_sqrt)
    return deg_mat @ a_tilde @ deg_mat


class GraphConvolutionLayer(nn.Module):
    """
    Standard single-layer Graph Convolution: H' = A @ H @ W + b
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, adj: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            adj: Normalized adjacency matrix (N, N).
            h: Node feature representations (N, in_features).
            
        Returns:
            Updated node representations (N, out_features).
        """
        support = self.linear(h)
        return torch.matmul(adj, support)


class VanillaGCN(nn.Module):
    """
    Two-layer Graph Convolutional Network that maps a graph's adjacency matrix
    and node features to a fixed-size 1D graph embedding via mean pooling.
    
    Formula:
        H^{(1)} = ReLU(A_hat * H^{(0)} * W^{(0)})
        H^{(2)} = ReLU(A_hat * H^{(1)} * W^{(1)})
        h_graph = MeanPool(H^{(2)})
    """

    def __init__(
        self,
        in_features: int = 16,
        hidden_dim: int = 64,
        out_features: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features

        self.gc1 = GraphConvolutionLayer(in_features, hidden_dim)
        self.gc2 = GraphConvolutionLayer(hidden_dim, out_features)
        self.dropout = nn.Dropout(dropout)

    def _generate_dummy_features(
        self,
        num_nodes: int,
        adj: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Generates deterministic, informative node features when none are provided.
        Combines normalized degree signals with unit initialization for anonymous DAGs.
        """
        x = torch.ones((num_nodes, self.in_features), device=device, dtype=dtype)
        # Inject degree information into the first two feature dimensions if available
        if self.in_features >= 2:
            out_degree = torch.sum(adj, dim=1, keepdim=True)
            in_degree = torch.sum(adj, dim=0, keepdim=True).t()
            x[:, 0:1] = out_degree / (num_nodes + 1e-6)
            x[:, 1:2] = in_degree / (num_nodes + 1e-6)
        return x / (self.in_features ** 0.5)

    def forward_single(
        self,
        adj: torch.Tensor,
        x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes a single graph into a 1D embedding vector.
        
        Args:
            adj: 2D Adjacency matrix of shape (N, N).
            x: Optional node feature matrix of shape (N, in_features).
            
        Returns:
            1D graph embedding tensor of shape (out_features,).
        """
        num_nodes = adj.size(0)
        device = adj.device
        dtype = adj.dtype

        if x is None:
            x = self._generate_dummy_features(num_nodes, adj, device, dtype)

        norm_adj = normalize_adjacency(adj)
        h1 = F.relu(self.gc1(norm_adj, x))
        h1 = self.dropout(h1)
        h2 = F.relu(self.gc2(norm_adj, h1))

        # Mean pooling across all nodes in the DAG
        graph_embedding = torch.mean(h2, dim=0)  # Shape: (out_features,)
        return graph_embedding

    def forward(
        self,
        adj_matrices: Union[torch.Tensor, List[torch.Tensor]],
        node_features: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """
        Encodes a batch of graphs into a 2D tensor of shape (batch_size, out_features).
        
        Args:
            adj_matrices: List of (N_i, N_i) tensors or a 3D tensor (B, N, N).
            node_features: Optional list of node features or 3D tensor.
            
        Returns:
            Tensor of shape (B, out_features).
        """
        if isinstance(adj_matrices, torch.Tensor) and adj_matrices.dim() == 2:
            # Single graph passed
            return self.forward_single(adj_matrices, node_features).unsqueeze(0)

        batch_embeddings: List[torch.Tensor] = []
        if isinstance(adj_matrices, list):
            for idx, adj in enumerate(adj_matrices):
                feat = node_features[idx] if node_features is not None else None
                emb = self.forward_single(adj, feat)
                batch_embeddings.append(emb)
        elif isinstance(adj_matrices, torch.Tensor) and adj_matrices.dim() == 3:
            # (B, N, N)
            for i in range(adj_matrices.size(0)):
                feat = node_features[i] if node_features is not None else None
                emb = self.forward_single(adj_matrices[i], feat)
                batch_embeddings.append(emb)
        else:
            raise ValueError(f"Unsupported adjacency input type/dim: {type(adj_matrices)}")

        return torch.stack(batch_embeddings, dim=0)


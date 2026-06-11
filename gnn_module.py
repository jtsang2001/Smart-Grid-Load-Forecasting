"""
gnn_module.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet


#Create an NxN matrix of ones, these will then get updated with the feature weights
def build_full_adjacency(n_nodes: int) -> torch.Tensor:
    return torch.ones(n_nodes, n_nodes, dtype=torch.float32)

# Thresholding correlation matrix
# def build_correlation_adjacency(
#     df,
#     feature_cols: list,
#     threshold: float = 0.3,
# ) -> torch.Tensor:
#     corr = df[feature_cols].corr().abs().values  # (N, N)
#     adj  = (corr >= threshold).astype(np.float32)
#     np.fill_diagonal(adj, 1.0)
#     return torch.tensor(adj, dtype=torch.float32)

#Gets the number of nodes
def get_encoder_cont_size(dataset: TimeSeriesDataSet) -> int:
    n = (
        len(dataset.time_varying_unknown_reals)
        + len(dataset.time_varying_known_reals)
    )
    if getattr(dataset, "add_relative_time_idx", False):
        n += 1
    if getattr(dataset, "add_encoder_length", False):
        n += 1
    if getattr(dataset, "add_target_scales", False):
        n += 2   # target mean + target std
    return n


# Multi-head GAT layer
class MultiHeadGATLayer(nn.Module):
    #Define the matrix that will project every node into attention space
    def __init__(
        self,
        in_dim:   int,
        out_dim:  int,
        n_heads:  int,
        dropout:  float = 0.1,
    ):
        super().__init__()
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        self.n_heads  = n_heads
        self.head_dim = out_dim // n_heads
        self.out_dim  = out_dim

        # Shared linear projection for all heads (1->32)
        self.W = nn.Linear(in_dim, out_dim, bias=False)

        self.attn_src = nn.Linear(self.head_dim, 1, bias=False)
        self.attn_dst = nn.Linear(self.head_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.act     = nn.ELU()

        self._reset_parameters()

    #Resets weights
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attn_src.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attn_dst.weight, gain=1.0)

    #Projects the node embeddings and resizes them into 4 seperate 8D heads
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        M, N, _ = h.shape

        # Linear transform → reshape to (M, N, H, head_dim = 8)
        #L6
        Wh = self.W(h).view(M, N, self.n_heads, self.head_dim)

        # Attention scores, colapses the embeddings into one scaler
        e_src = self.attn_src(Wh)                   
        e_dst = self.attn_dst(Wh)                   
        e = e_src.unsqueeze(2) + e_dst.unsqueeze(1) 
        e = F.leaky_relu(e.squeeze(-1), 0.2)        

        #Set non-edges to 0 after softmax so nothing is contributed to aggregation
        mask = (adj == 0).unsqueeze(0).unsqueeze(-1)
        e = e.masked_fill(mask, float("-inf"))

        # Softmax over incoming neighbours
        alpha = F.softmax(e, dim=2)                 
        alpha = self.dropout(alpha)

        # Weighted aggregation: sum over neighbours, to aggregate back to 32D from the four 8D heads
        out = (alpha.unsqueeze(-1) * Wh.unsqueeze(1)).sum(dim=2)
        out = out.reshape(M, N, self.out_dim)

        return self.act(out)


# GAT-TFT Encoder
class FeatureGATEncoder(nn.Module):
    #Define the GAT Layers
    def __init__(
        self,
        n_nodes:    int,
        hidden_dim: int   = 32,
        n_heads:    int   = 4,
        n_layers:   int   = 2,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % n_heads == 0

        self.n_nodes = n_nodes

        # Project scalar node value → hidden_dim (32D)
        # L5
        self.input_proj = nn.Linear(1, hidden_dim)

        # Stack of GAT layers with layer norm + residual
        self.gat_layers = nn.ModuleList([
            MultiHeadGATLayer(hidden_dim, hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        #L7 L9
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(n_layers)
        ])

        # Project back to 1 scalar per node (32D -> 1D)
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Small gate to control how much the GNN correction contributes
        # initialised to 0 so training starts gradually opens up
        self.gate = nn.Parameter(torch.zeros(1))


    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, T, N = x.shape

        # Reshapes the batch and window together to become independent graph instance
        h = x.reshape(B * T, N, 1)
        #Project to 32D         
        h = self.input_proj(h)              

        #Perform two rounds of message passing (2 layers) and normalise the results
        #L6 L8
        for gat, norm in zip(self.gat_layers, self.layer_norms):
            h = norm(h + gat(h, adj))

        # Collapses 32D back to 1D
        h = self.output_proj(h)             
        h = h.squeeze(-1).reshape(B, T, N)

        # Run results (h) through tanh gate to scale GAT output
        #L10
        return torch.tanh(self.gate) * h

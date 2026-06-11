"""
model.py
"""

#Imports
import torch
import torch.nn as nn
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from gnn_module import FeatureGATEncoder, build_full_adjacency, get_encoder_cont_size

class GNNEnhancedTFT(TemporalFusionTransformer):
    #Define GAT Parameters
    def __init__(
        self,
        adj_matrix:  torch.Tensor,
        gnn_hidden:  int   = 32,
        gnn_heads:   int   = 4,
        gnn_layers:  int   = 2,
        gnn_dropout: float = 0.1,
        **tft_kwargs,
    ):
        #Calls TFT constructor and attach the TFT as a submodule
        super().__init__(**tft_kwargs)

        n_nodes = int(adj_matrix.shape[0])

        self.register_buffer("_adj", adj_matrix.float())

        # Build the GAT encoder
        self.gat = FeatureGATEncoder(
            n_nodes    = n_nodes,
            hidden_dim = gnn_hidden,
            n_heads    = gnn_heads,
            n_layers   = gnn_layers,
            dropout    = gnn_dropout,
        )

    #Read the dataset and pass in the GAT Parameters
    @classmethod
    def from_dataset(
        cls,
        dataset:     TimeSeriesDataSet,
        adj_matrix:  torch.Tensor,
        gnn_hidden:  int   = 32,
        gnn_heads:   int   = 4,
        gnn_layers:  int   = 2,
        gnn_dropout: float = 0.1,
        **kwargs,
    ) -> "GNNEnhancedTFT":
        
        return super().from_dataset(
            dataset,
            adj_matrix  = adj_matrix,
            gnn_hidden  = gnn_hidden,
            gnn_heads   = gnn_heads,
            gnn_layers  = gnn_layers,
            gnn_dropout = gnn_dropout,
            **kwargs,
        )

    # Create a dictionary to contain the information that will get processed through the GAT to enrich it
    # that will then get passed to the TFT to read
    def forward(self, x: dict) -> dict:

        enc = x["encoder_cont"]                     # (B, T, N)
        residual = self.gat(enc, self._adj)          # (B, T, N)
        x = {**x, "encoder_cont": enc + residual}

        return super().forward(x)

#Build the enhanced GAT-TFT model
def build_gnn_tft(
    dataset:    TimeSeriesDataSet,
    *,
    gnn_hidden:  int   = 32,
    gnn_heads:   int   = 4,
    gnn_layers:  int   = 2,
    gnn_dropout: float = 0.1,
    # TFT hyperparameters
    learning_rate:          float = 3e-4,
    hidden_size:            int   = 160,
    attention_head_size:    int   = 4,
    dropout:                float = 0.1,
    hidden_continuous_size: int   = 80,
    reduce_on_plateau_patience: int = 4,
    **extra_tft_kwargs,
) -> GNNEnhancedTFT:

    n_nodes = get_encoder_cont_size(dataset)
    adj     = build_full_adjacency(n_nodes)
    #Build the model with the specified parameters above
    model = GNNEnhancedTFT.from_dataset(
        dataset,
        adj_matrix                 = adj,
        gnn_hidden                 = gnn_hidden,
        gnn_heads                  = gnn_heads,
        gnn_layers                 = gnn_layers,
        gnn_dropout                = gnn_dropout,
        learning_rate              = learning_rate,
        hidden_size                = hidden_size,
        attention_head_size        = attention_head_size,
        dropout                    = dropout,
        hidden_continuous_size     = hidden_continuous_size,
        loss                       = QuantileLoss(),
        optimizer                  = "adamw",
        reduce_on_plateau_patience = reduce_on_plateau_patience,
        log_interval               = 10,
        log_val_interval           = 1,
        **extra_tft_kwargs,
    )
    return model

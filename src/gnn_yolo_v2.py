"""
GNN module for post-processing YOLO aneurysm detections - Binary Classification Version.

Takes YOLO predictions and constructs a spatial graph where nodes represent
detected bounding boxes with features: confidence, spatial position, and class.
Outputs binary prediction: aneurysm present/absent.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import math


class SimpleGNNBinary(nn.Module):
    """
    Simple GNN for binary aneurysm detection (present/absent).

    Node features:
        [confidence, x_center, y_center, z_norm, z_mm_norm, class_one_hot]
    (When z is unavailable, set z_norm and z_mm_norm to 0.)
    Edge features: spatial/physical distance-based weights
    """

    def __init__(self, num_classes: int = 13, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Node feature dimension: conf(1) + coords(4: x,y,z,z_mm) + class_onehot(13) = 18
        self.node_feat_dim = 1 + 4 + num_classes

        # Node embedding
        self.node_embed = nn.Linear(self.node_feat_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Binary classification head: aneurysm present/absent
        self.binary_head = nn.Linear(hidden_dim, 1)  # Series-level binary classification
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, node_feat_dim] 
            edge_index: [2, num_edges] adjacency
            edge_weights: [num_edges] optional edge weights
            
        Returns:
            dict with binary_logit
        """
        # Embed nodes
        x = F.relu(self.node_embed(node_features))
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_weights)

        # Global pooling to get series-level representation
        if x.size(0) > 0:
            # Max pooling across all nodes to get series representation
            series_repr = torch.max(x, dim=0)[0]  # [hidden_dim]
        else:
            # Handle empty graphs
            series_repr = torch.zeros(self.hidden_dim, device=x.device)
        
        # Series-level binary prediction
        binary_logit = self.binary_head(series_repr)  # [1]
        
        return {
            'binary_logit': binary_logit.squeeze(-1)  # scalar
        }


class GNNLayer(nn.Module):
    """Single GNN layer with attention-weighted message passing."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Attention mechanism for confidence-aware message passing
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention-weighted message passing layer."""
        row, col = edge_index
        
        # Compute messages
        messages = self.message_mlp(torch.cat([x[row], x[col]], dim=-1))
        
        # Compute attention weights based on node features
        attention_input = torch.cat([x[row], x[col]], dim=-1)
        attention_weights = torch.sigmoid(self.attention(attention_input)).squeeze(-1)
        
        # Combine edge weights with attention weights
        if edge_weights is not None:
            final_weights = edge_weights * attention_weights
        else:
            final_weights = attention_weights
        
        # Apply combined weights to messages
        messages = messages * final_weights.unsqueeze(-1)
        
        # Aggregate messages for each node with normalization
        num_nodes = x.size(0)
        aggregated = torch.zeros(num_nodes, self.hidden_dim, device=x.device)
        norm_weights = torch.zeros(num_nodes, device=x.device)
        
        aggregated.index_add_(0, col, messages)
        norm_weights.index_add_(0, col, final_weights)
        
        # Normalize by sum of weights (avoid division by zero)
        norm_weights = torch.clamp(norm_weights, min=1e-6)
        aggregated = aggregated / norm_weights.unsqueeze(-1)
        
        # Update node features
        x_new = self.update_mlp(torch.cat([x, aggregated], dim=-1))
        return F.relu(x_new + x)  # Residual connection



def create_binary_gnn_model(num_classes: int = 13, hidden_dim: int = 64) -> SimpleGNNBinary:
    """Factory function to create binary GNN model."""
    return SimpleGNNBinary(num_classes=num_classes, hidden_dim=hidden_dim)



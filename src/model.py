import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GraphSAGE, GAT
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_max_pool, global_mean_pool

class GraphModel(nn.Module):
    def __init__(self, use_pe,
                 walk_length,
                 num_layers,
                 hidden_channels,
                 jk,
                 dropout):
        super().__init__()
        self.gnn = GraphSAGE(
            256 + walk_length if use_pe else 256,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # not final class count yet
            jk=jk,
            dropout=dropout,
            norm=LayerNorm(hidden_channels),
        )
        self.cls = nn.Sequential(
            nn.Linear(hidden_channels, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 14)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Move to CUDA
        x = x.cuda()
        edge_index = edge_index.cuda() if edge_index.numel() > 0 else torch.tensor([[0, 0]]).T.cuda()
        batch = batch.cuda()

        # Get node embeddings
        node_embeddings = self.gnn(x, edge_index, batch=batch)

        # Pool to graph embeddings
        graph_embeddings = global_max_pool(node_embeddings, batch)
        # or: global_mean_pool(node_embeddings, batch)

        # Classify graphs
        logits = self.cls(graph_embeddings)
        return logits[:, :1], logits[:, 1:]



class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""

    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()

        self.model_name = model_name

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,  # Remove classifier head
            global_pool=''  # Remove global pooling
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chans, img_size, img_size)
            features = self.backbone(dummy_input)

            if len(features.shape) == 4:
                # Conv features (batch, channels, height, width)
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                # Transformer features (batch, sequence, features)
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # Already flat features (batch, features)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False

        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")

        # Add global pooling for models that output spatial features
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Combined classifier with batch norm for stability
        self.loc_classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        self.aneurysm_classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 1)
        )

    def forward(self, image):
        # Extract image features
        img_features = self.backbone(image)

        # Apply appropriate pooling based on model type
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # Conv features - apply global pooling
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer features - average across sequence dimension
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            # Fallback for any 4D output
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            # Fallback for any 3D output
            img_features = img_features.mean(dim=1)

        # Classification
        loc_output = self.loc_classifier(img_features)
        cls_logit = self.aneurysm_classifier(img_features)

        return cls_logit, loc_output

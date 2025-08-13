import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GraphSAGE, GAT
from torch_geometric.nn import LayerNorm
from torch_geometric.nn import global_max_pool, global_mean_pool

class GraphModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gnn = GraphSAGE(
            256 + cfg.walk_length if cfg.use_pe else 256,
            num_layers=cfg.num_layers,
            hidden_channels=cfg.hidden_channels,
            out_channels=cfg.hidden_channels,  # not final class count yet
            jk=cfg.jk,
            dropout=cfg.dropout,
            norm=LayerNorm(cfg.hidden_channels),
        )
        # Final classification head
        self.fc = nn.Linear(cfg.hidden_channels, 14)  # 14 graph classes

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
        logits = self.fc(graph_embeddings)
        return logits[:, :1], logits[:, 1:]



class MultiBackboneModel(nn.Module):

    def __init__(self, model_name, in_chans, img_size, num_classes=13, pretrained=True,
                 drop_rate=0.3, drop_path_rate=0.2,
                 global_pool_override: str | None = None,
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.img_size = img_size
        self.original_in_chans = in_chans

        # Build kwargs for backbone construction
        backbone_kwargs = dict(
            pretrained=pretrained,
            in_chans=in_chans,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            num_classes=0,
            img_size=img_size,
        )
        # Some models (e.g., DeiT) assert a specific global_pool ('token'), so pass override only when specified
        if global_pool_override is not None:
            backbone_kwargs["global_pool"] = global_pool_override

        try:
            self.backbone = timm.create_model(model_name, **backbone_kwargs)
        except TypeError:
            backbone_kwargs.pop("img_size", None)
            self.backbone = timm.create_model(model_name, **backbone_kwargs)

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

     

        # Combined classifier with batch norm for stability (location)
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

        img_features = self.backbone(image)

        # Apply appropriate pooling based on model type
        if hasattr(self, 'needs_pool') and self.needs_pool:
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            img_features = img_features.mean(dim=1)

        # Classification heads
        loc_output = self.loc_classifier(img_features)
        cls_logit = self.aneurysm_classifier(img_features)

        return cls_logit, loc_output


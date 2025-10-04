import timm
import torch.nn as nn
from torch_geometric.nn.models import GraphSAGE
import torch
from torch_geometric.nn import LayerNorm, global_max_pool
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE, global_max_pool, global_mean_pool
from torch_geometric.nn.norm import LayerNorm

class GraphModel(nn.Module):
    def __init__(self, hidden_channels=256, num_layers=8, jk='lstm',
                 walk_length=8, use_pe=True, dropout=0.3, pooling="noisy_or", tau=0.1):
        super().__init__()
        in_dim = 256 + walk_length if use_pe else 256
        self.gnn = GraphSAGE(
            in_channels=in_dim,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            out_channels=1,
            jk=jk,
            dropout=dropout,
            norm=LayerNorm(in_channels=hidden_channels)
        )

        self.pooling = pooling
        self.tau = tau  # temperature for logsumexp pooling

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        edge_index = edge_index.cuda()
        batch = batch.cuda()
        x = x.cuda()

        if edge_index.shape[0] == 0:
            edge_index = torch.tensor([[0, 0]]).T.cuda()

        # Node logits
        node_logits = self.gnn(x, edge_index, batch=batch)  # (N, 1)

        # Graph-level pooling
        if self.pooling == "max":
            graph_logits = global_max_pool(node_logits, batch)

        elif self.pooling == "mean":
            graph_logits = global_mean_pool(node_logits, batch)

        elif self.pooling == "noisy_or":
            node_probs = torch.sigmoid(node_logits)
            # Graph prob = 1 - product(1 - p_i)
            graph_probs = 1 - global_mean_pool(1 - node_probs, batch)
            graph_logits = torch.logit(torch.clamp(graph_probs, 1e-6, 1 - 1e-6))

        elif self.pooling == "logsumexp":
            # smooth max pooling
            # group nodes by graph
            num_graphs = batch.max().item() + 1
            graph_logits = []
            for g in range(num_graphs):
                mask = batch == g
                if mask.any():
                    scores = node_logits[mask] / self.tau
                    pooled = self.tau * torch.logsumexp(scores, dim=0)
                    graph_logits.append(pooled)
                else:
                    graph_logits.append(torch.tensor([0.0], device=node_logits.device))
            graph_logits = torch.stack(graph_logits, dim=0)

        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

        return node_logits, graph_logits



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
        #self.loc_classifier = nn.Sequential(
        #    nn.Linear(num_features, 512),
        #    nn.BatchNorm1d(512),
        #    nn.ReLU(),
        #    nn.Dropout(drop_rate),
        #    nn.Linear(512, 256),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(),
        #    nn.Dropout(drop_rate),
        #    nn.Linear(256, num_classes)
        #)
#
#
        #self.aneurysm_classifier = nn.Sequential(
        #    nn.Linear(num_features, 256),
        #    nn.BatchNorm1d(256),
        #    nn.ReLU(),
        #    nn.Dropout(drop_rate),
        #    nn.Linear(256, 128),
        #    nn.BatchNorm1d(128),
        #    nn.ReLU(),
        #    nn.Dropout(drop_rate),
        #    nn.Linear(128, 1)
        #)
#
        self.loc_classifier = nn.Linear(num_features, num_classes)
        self.aneurysm_classifier = nn.Linear(num_features, 1)
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


import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class AttentionMIL(nn.Module):
    """Attention pooling from Ilse et al., 2018 (https://arxiv.org/abs/1802.04712)."""
    def __init__(self, in_dim, out_dim, hidden_dim=256):
        super().__init__()
        self.attention_V = nn.Linear(in_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, 1)
        self.classifier = nn.Linear(in_dim, out_dim)

    def forward(self, features):
        # features: (B, S, D)
        A = torch.tanh(self.attention_V(features))     # (B, S, H)
        A = self.attention_U(A)                        # (B, S, 1)
        A = torch.softmax(A, dim=1)                    # attention weights
        M = torch.sum(A * features, dim=1)             # weighted sum: (B, D)
        return self.classifier(M), A.squeeze(-1)


class AneurysmClassifier(nn.Module):
    def __init__(self, num_classes=13, slice_aux=True):
        super().__init__()
        
        # Backbone encoder: ResNet18, remove classification head
        base = timm.create_model("resnet18", pretrained=True, in_chans=3, num_classes=0, global_pool='avg')
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # output (B, 512, 1, 1)
        self.feature_dim = 512

        # MIL aggregator
        self.cls_attention = AttentionMIL(self.feature_dim, 1)
        self.loc_attention = AttentionMIL(self.feature_dim, num_classes)

        # Auxiliary slice-level classifier
        self.slice_aux = slice_aux
        if slice_aux:
            self.slice_classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, slice_labels=None):
        """
        x: (B, S, C, H, W)
        slice_labels: (B, S, num_classes) with 1 for aneurysm slice, else 0
        """
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        # Extract slice embeddings
        feats = self.encoder(x)       # (B*S, 512, 1, 1)
        feats = feats.view(B, S, -1)  # (B, S, 512)

        cls_logits, _ = self.cls_attention(feats)
        loc_logits, _ = self.loc_attention(feats)

        output = {
            "cls_logits": cls_logits.squeeze(),
            "loc_logits": loc_logits,
        }
        if self.slice_aux and slice_labels is not None:
            slice_logits = self.slice_classifier(feats)  # (B, S, num_classes)
            # BCE per slice, masked only on slices that actually contain aneurysm labels
            mask = slice_labels.sum(-1) > 0  # (B, S)
            if mask.any():
                aux_loss = F.binary_cross_entropy_with_logits(
                    slice_logits[mask], slice_labels[mask].float()
                )
                output['aux_loss'] = aux_loss
        return output


import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""
    def __init__(self, model_name, num_classes, pretrained, in_chans, drop_rate, drop_path_rate):
        super().__init__()
        '''
        2D: Input volume with size Depth(as channel) x Height x Width -> dxhxw feature map
        3D: expand channel to 1xdxhxw -> lstm along depth -> cls predictions
        '''
        self.model_name = model_name
        self.loc_classifier = timm.create_model(
             model_name,
             num_classes=num_classes,
             pretrained=pretrained,
             in_chans=in_chans,
            drop_rate= drop_rate,
            drop_path_rate= drop_path_rate
        )

    def forward(self, image):
        return self.loc_classifier(image)

# ------------------------
# ConvGRUCell
# ------------------------
class ConvGRUCell(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, kernel_size=(3, 3), bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = None

    def build_conv(self, in_ch, hidden_ch):
        padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        return nn.Conv2d(in_ch + hidden_ch, 3 * hidden_ch,
                         kernel_size=self.kernel_size,
                         padding=padding,
                         bias=self.bias)

    def forward(self, x, h_cur):
        in_ch = x.shape[1]
        hidden_ch = h_cur.shape[1]
        if self.conv is None or self.conv.in_channels != in_ch + hidden_ch:
            self.conv = self.build_conv(in_ch, hidden_ch).to(x.device)

        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_z, cc_r, cc_h = torch.split(conv_out, hidden_ch, dim=1)

        z = torch.sigmoid(cc_z)
        r = torch.sigmoid(cc_r)
        h_tilde = torch.tanh(cc_h + r * h_cur)
        h_next = (1 - z) * h_cur + z * h_tilde
        return h_next

# ------------------------
# ConvGRU wrapper
# ------------------------
class ConvGRU(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, kernel_size=(3, 3),
                 num_layers=1, batch_first=True):
        super().__init__()
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size] * len(self.hidden_dim)
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.cells = nn.ModuleList([ConvGRUCell(
            input_dim=None if i == 0 else self.hidden_dim[i - 1],
            hidden_dim=self.hidden_dim[i],
            kernel_size=self.kernel_size[i]
        ) for i in range(self.num_layers)])

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        B, T, C, H, W = x.shape

        if hidden_state is None:
            hidden_state = [torch.zeros(B, hdim, H, W, device=x.device) for hdim in self.hidden_dim]

        layer_outputs = []
        seq = x
        for l, cell in enumerate(self.cells):
            h = hidden_state[l]
            outputs = []
            for t in range(T):
                h = cell(seq[:, t], h)
                outputs.append(h)
            seq = torch.stack(outputs, dim=1)
            layer_outputs.append(seq)
        return layer_outputs, hidden_state

# ------------------------
# MultiBackboneModel with ConvGRU
# ------------------------
class ConvGRUModel(nn.Module):
    def __init__(self,
                 model_name="tf_efficientnetv2_s.in21k_ft_in1k",
                 num_classes=14,
                 pretrained=True,
                 drop_rate=0.3,
                 drop_path_rate=0.2,
                 convgru_hidden_dim=64,
                 reduce_seq_len=16,
                 channel_reduce_out=16):   # keep >1 channels per timestep if wanted
        super().__init__()
        self.reduce_seq_len = reduce_seq_len
        self.channel_reduce_out = channel_reduce_out

        # 2D CNN backbone (slice-wise)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=8,        # process slices independently
            num_classes=0,
            global_pool="",
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # Depth reduction: convolution along depth axis only
        self.depth_reduce = nn.Conv3d(
            in_channels=1280,               # backbone output channels
            out_channels=channel_reduce_out,
            kernel_size=(3, 1, 1),          # only across depth
            stride=(max(1, 1280 // reduce_seq_len), 1, 1),
            padding=(1, 0, 0)
        )

        # ConvGRU operates on sequence (T timesteps = reduced depth)
        self.convgru = ConvGRU(
            input_dim=channel_reduce_out,
            hidden_dim=convgru_hidden_dim,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        # Final classifier
        self.fc = nn.Linear(convgru_hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (B, D, H, W) volume
        """
        B, _, H, W = x.shape
        D = 4

        #4 depth chunks, each with 8 depth window size
        #eff->mip
        x_slices = x.reshape(B * D, 32//D, H, W)          # (B*D, 1, H, W)
        feat = self.backbone(x_slices)                # (B*D, C, Hf, Wf)
        _, C, Hf, Wf = feat.shape
        feat = feat.view(B, D, C, Hf, Wf)             # (B, D, C, Hf, Wf)

        # Move depth (D) into 3D conv input format
        feat = feat.permute(0, 2, 1, 3, 4)            # (B, C, D, Hf, Wf)

        # Depth reduction conv
        feat = self.depth_reduce(feat)                # (B, Cout, D', H, Wf)

        # Prepare sequence for ConvGRU
        feat = feat.permute(0, 2, 1, 3, 4)            # (B, T=D', C, Hf, Wf)

        # ConvGRU
        gru_out, _ = self.convgru(feat)
        last_hidden = gru_out[0][:, -1]              # (B, hidden_dim, Hf, Wf)

        # Pool + FC
        pooled = F.adaptive_avg_pool2d(last_hidden, 1).flatten(1)
        return self.fc(pooled)

# ------------------------
# MultiViewPatchModel
# ------------------------
class AttentionPool(nn.Module):
    """Attention pooling across a token dimension.
       Input: x (B, n_tokens, D)
       Output: pooled (B, D) or (B, n_groups, D) if grouped beforehand.
    """
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: (B, T, D)
        scores = self.score(x).squeeze(-1)  # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        pooled = (x * attn).sum(dim=1)  # (B, D)
        return pooled, attn.squeeze(-1)  # return pooled and weights for debugging


class MultiViewPatchModel(nn.Module):
    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        drop_rate: float = 0.3,
        drop_path_rate: float = 0.2,
        k_candi: int = 3,
        model_dim: int = 512,
        transformer_layers: int = 2,
        transformer_heads: int = 8,
        classifier_hidden: int = 256,
        dropout: float = 0.2,
    ):
        """
        Args:
            model_name: timm model name used for each view backbone
            k_candi: number of candidates per image (Top-K)
            model_dim: final shared embedding dimension (transformer d_model)
        """
        super().__init__()
        self.k_candi = k_candi
        self.model_keys = [
            "axial_mip",
            "sagittal_mip",
            "coronal_mip",
            "axial_lp",
            "sagittal_lp",
            "coronal_lp",
            "axial_vol",
            "sagittal_vol",
            "coronal_vol",
        ]
        # in_chans for these 9 backbones: first 6 are 1-channel MIPs/LPs, last 3 are volumetric with 31 channels
        in_chans_list = [1] * 6 + [31] * 3
        assert len(in_chans_list) == len(self.model_keys)

        # Create backbone dict
        self.backboneDict = nn.ModuleDict()
        for key, in_ch in zip(self.model_keys, in_chans_list):
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                in_chans=in_ch,
                num_classes=0,       # return features
                global_pool="",
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
            )
            self.backboneDict[key] = backbone

        # Determine backbone output dim (try common attributes then fallback)
        # We'll assume all backbones produce the same last feature dim (common for timm models)
        sample_backbone = next(iter(self.backboneDict.values()))
        out_dim = getattr(sample_backbone, "num_features", None)
        if out_dim is None:
            out_dim = getattr(sample_backbone, "embed_dim", None)
        if out_dim is None:
            # fallback
            out_dim = 512

        self.backbone_out_dim = out_dim
        self.model_dim = model_dim

        # Project each backbone output to shared model_dim
        self.project = nn.Linear(self.backbone_out_dim, self.model_dim)

        # Positional embedding for (9 * k_candi) tokens
        self.seq_len = 9 * self.k_candi
        self.pos_emb = nn.Parameter(torch.zeros(1, self.seq_len, self.model_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, nhead=transformer_heads, dim_feedforward=self.model_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Attention pooling to aggregate 9 view tokens per candidate
        self.attn_pool = AttentionPool(self.model_dim)

        # Classifier head (per candidate -> binary logit)
        self.classifier = nn.Sequential(
            nn.Linear(self.model_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 1),
        )

        # small init
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _flatten_input_for_backbone(self, x):
        """
        Accept either:
         - x shape = (B, K, C, H, W)  -> flatten to (B*K, C, H, W)
         - x shape = (B*K, C, H, W)    -> keep as is
        """
        if x.dim() == 5:
            B, K, C, H, W = x.shape
            x_flat = x.view(B * K, C, H, W)
            return x_flat, B
        elif x.dim() == 4:
            # caller passed already flattened
            # cannot infer B directly; assume they pass B*K as first dim -> user must ensure consistency
            return x, None
        else:
            raise ValueError("Unsupported input tensor shape for view patch. Expect 4D or 5D tensor.")

    def forward(self, patch: dict):
        """
        patch: dict[str -> tensor] mapping each key in self.model_keys to a tensor shaped:
               (B, K, C, H, W)  OR  (B*K, C, H, W)
               Order of keys must match self.model_keys.

        Returns:
            logits: (B, K) tensor of logits for each candidate in the batch.
        """
        device = next(self.parameters()).device

        # Collect features per key
        feats_per_key = []
        inferred_B = None
        for key in self.model_keys:
            if key not in patch:
                raise KeyError(f"Missing expected key '{key}' in patch dict.")
            x = patch[key].to(device)
            x_flat, maybe_B = self._flatten_input_for_backbone(x)
            if inferred_B is None and maybe_B is not None:
                inferred_B = maybe_B
            # pass through backbone: many timm models when num_classes=0 return (N, feat_dim)
            feat = self.backboneDict[key](x_flat)  # (B*K, backbone_out_dim)
            if feat.dim() == 4:
                # some backbones may return (N, C, 1, 1) depending - flatten
                feat = feat.view(feat.size(0), -1)
            feats_per_key.append(feat)

        if inferred_B is None:
            # If user passed flattened inputs, attempt to infer B from self.k_candi
            # First backbone feature's first dim is expected to be B*K
            N_flat = feats_per_key[0].size(0)
            if (N_flat % self.k_candi) != 0:
                raise ValueError("Cannot infer batch size from flattened inputs and k_candi. Ensure inputs shaped (B*K, C, H, W) or pass B*K divisible by k_candi.")
            inferred_B = N_flat // self.k_candi

        B = inferred_B
        K = self.k_candi
        # Stack features: list length = 9 ; each tensor shape = (B*K, out_dim)
        stacked = torch.stack(feats_per_key, dim=1)  # (B*K, 9, out_dim)

        # reshape to (B, K, 9, out_dim)
        stacked = stacked.view(B, K, len(self.model_keys), self.backbone_out_dim)  # (B, K, 9, out_dim)

        # reorder to tokens grouped by candidate: (B, K*9, out_dim)
        tokens = stacked.view(B, K * len(self.model_keys), self.backbone_out_dim)  # (B, 9*K, out_dim)

        # project features
        tokens = self.project(tokens)  # (B, 9*K, model_dim)

        # add pos emb (pos_emb length must equal seq_len which is 9*K)
        if tokens.size(1) != self.seq_len:
            # If user passed K different than initialized, handle by slicing/expanding pos emb
            if tokens.size(1) < self.seq_len:
                pos = self.pos_emb[:, : tokens.size(1), :].to(device)
            else:
                # expand pos emb by repeating
                repeats = int(tokens.size(1) / self.seq_len)
                pos = self.pos_emb.repeat(1, repeats + 1, 1)[:, : tokens.size(1), :].to(device)
        else:
            pos = self.pos_emb.to(device)
        tokens = tokens + pos

        # transformer expects (B, S, E) since we set batch_first=True
        tokens = self.transformer(tokens)  # (B, 9*K, model_dim)

        # reshape to (B, K, 9, model_dim) then pool across the 9 views per candidate
        tokens_per_candidate = tokens.view(B, K, len(self.model_keys), self.model_dim)  # (B, K, 9, D)

        # We'll pool across dim=2 (the 9 views)
        # Perform attention pooling for each candidate
        # flatten first two dims to run pooling in one go
        tokens_flat = tokens_per_candidate.view(B * K, len(self.model_keys), self.model_dim)  # (B*K, 9, D)
        pooled, attn_weights = self.attn_pool(tokens_flat)  # pooled: (B*K, D)
        pooled = pooled.view(B, K, self.model_dim)  # (B, K, D)

        # classify each candidate
        logits = self.classifier(pooled)  # (B, K, 1)
        logits = logits.squeeze(-1)  # (B, K)

        return logits  # logits per candidate

# === Example usage ===
if __name__ == "__main__":
    # Example instantiation:
    model = MultiViewPatchModel(model_name="resnet18",
                                pretrained=False,
                                k_candi=3,
                                model_dim=512,
                                transformer_layers=1)

    # Example fake input tensors for one batch:
    B = 2
    K = 3
    H, W = 128, 128

    patch = {}
    # first six keys are 1-channel 2D patches shaped (B, K, 1, H, W)
    for key in model.model_keys[:6]:
        patch[key] = torch.randn(B, K, 1, H, W)
    # last three keys are volume patches with 31 channels shaped (B, K, 31, H, W)
    for key in model.model_keys[6:]:
        patch[key] = torch.randn(B, K, 31, H, W)

    logits = model(patch)  # (B, K)
    print("logits", logits.shape)
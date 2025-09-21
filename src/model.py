import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# class MultiBackboneModel(nn.Module):
#     """Flexible model that can use different backbones"""
#     def __init__(self, model_name, **kwargs):
#         super().__init__()
#         '''
#         2D: Input volume with size Depth(as channel) x Height x Width -> dxhxw feature map
#         3D: expand channel to 1xdxhxw -> lstm along depth -> cls predictions
#         '''
#         self.model_name = model_name
#         self.loc_classifier = timm.create_model(
#              model_name,
#              num_classes=num_classes,
#              pretrained=pretrained,
#              in_chans=in_chans,
#             drop_rate= drop_rate,
#             drop_path_rate= drop_path_rate
#         )
#
#     def forward(self, image):
#         return self.loc_classifier(image)

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
class MultiBackboneModel(nn.Module):
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

        # Process each slice through backbone
        x_slices = x.reshape(B * 4, 8, H, W)          # (B*D, 1, H, W)
        feat = self.backbone(x_slices)                # (B*D, C, Hf, Wf)
        _, C, Hf, Wf = feat.shape
        feat = feat.view(B, D, C, Hf, Wf)             # (B, D, C, Hf, Wf)

        # Move depth (D) into 3D conv input format
        feat = feat.permute(0, 2, 1, 3, 4)            # (B, C, D, Hf, Wf)

        # Depth reduction conv
        feat = self.depth_reduce(feat)                # (B, Cout, D', Hf, Wf)

        # Prepare sequence for ConvGRU
        feat = feat.permute(0, 2, 1, 3, 4)            # (B, T=D', C, Hf, Wf)

        # ConvGRU
        gru_out, _ = self.convgru(feat)
        last_hidden = gru_out[0][:, -1]              # (B, hidden_dim, Hf, Wf)

        # Pool + FC
        pooled = F.adaptive_avg_pool2d(last_hidden, 1).flatten(1)
        return self.fc(pooled)
# ------------------------
# Quick unit test
# ------------------------
if __name__ == "__main__":
    model = MultiBackboneModel(num_classes=14,
                               reduce_seq_len=128,
                               convgru_hidden_dim=64,
                               channel_reduce_out=128)  # try 8 features per timestep

    x = torch.randn(2, 32, 384, 384)  # (B=2, depth=32, H=128, W=128)
    out = model(x)
    print("Output:", out.shape)  # should be (2, 13)
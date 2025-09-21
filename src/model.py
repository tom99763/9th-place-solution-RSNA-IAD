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
# ConvLSTM Cell
# ------------------------
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=256, kernel_size=(3,3), bias=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        self.conv = None
        self.input_dim = input_dim  # may be None initially

    def build_conv(self, input_dim, hidden_dim, kernel_size):
        padding = kernel_size[0] // 2, kernel_size[1] // 2
        return nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )

    def forward(self, x, h_cur, c_cur):
        # Build conv if not built yet or input channels changed
        if self.conv is None or x.shape[1] + h_cur.shape[1] != self.conv.in_channels:
            self.conv = self.build_conv(x.shape[1], h_cur.shape[1], self.kernel_size).to(x.device)

        combined = torch.cat([x, h_cur], dim=1)
        conv_out = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(conv_out, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

# ------------------------
# ConvLSTM Module
# ------------------------
class ConvLSTM(nn.Module):
    """ConvLSTM module for sequence input: x (B, T, C, H, W)"""
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, batch_first=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if isinstance(hidden_dim, list) else [hidden_dim]
        self.kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]*len(self.hidden_dim)
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i-1]
            self.cells.append(ConvLSTMCell(cur_input_dim, self.hidden_dim[i], self.kernel_size[i]))

    def forward(self, x, hidden_state=None):
        if not self.batch_first:
            x = x.permute(1, 0, 2, 3, 4)
        B, T, C, H, W = x.shape

        if hidden_state is None:
            hidden_state = []
            for i in range(self.num_layers):
                h = torch.zeros(B, self.hidden_dim[i], H, W, device=x.device)
                c = torch.zeros(B, self.hidden_dim[i], H, W, device=x.device)
                hidden_state.append((h, c))

        layer_output_list = []
        seq_input = x
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(T):
                h, c = self.cells[layer_idx](seq_input[:, t, :, :, :], h, c)
                output_inner.append(h)
            seq_input = torch.stack(output_inner, dim=1)  # (B, T, hidden, H, W)
            layer_output_list.append(seq_input)

        return layer_output_list, hidden_state

# ------------------------
# MultiBackboneModel
# ------------------------
class MultiBackboneModel(nn.Module):
    def __init__(self,
                 model_name="tf_efficientnetv2_s.in21k_ft_in1k",
                 num_classes=14,
                 in_chans=32,
                 pretrained=True,
                 drop_rate=0.3,
                 drop_path_rate=0.2,
                 convlstm_hidden_dim=64,
                 reduce_seq_len=64,
                 reduce_depth=True):
        super().__init__()

        # 2D CNN backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool=""
        )

        self.reduce_seq_len = reduce_seq_len
        self.reduce_depth = reduce_depth

        # placeholder for channel reduction
        self.channel_reduce = None

        # ConvLSTM: channels as sequence
        self.convlstm = ConvLSTM(
            input_dim=1,  # after reducing channels
            hidden_dim=convlstm_hidden_dim,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(convlstm_hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (B, D, H, W) - input volume
        """
        feat = self.backbone(x)  # (B, C, H', W')
        B, C, H, W = feat.shape

        # reduce depth (sequence length) if too long
        if self.reduce_depth and C > self.reduce_seq_len:
            feat = feat[:, :self.reduce_seq_len, :, :]
            C = feat.shape[1]

        # define channel reduction conv dynamically
        if self.channel_reduce is None or self.channel_reduce.in_channels != C:
            self.channel_reduce = nn.Conv2d(C, 1, kernel_size=1).to(feat.device)

        feat = self.channel_reduce(feat)  # (B, 1, H, W)

        # treat channels as sequence for ConvLSTM
        seq = feat.permute(0, 2, 3, 1).unsqueeze(1)  # (B, T=1, C=1, H, W)
        lstm_out, _ = self.convlstm(seq)  # list of outputs
        last_hidden = lstm_out[0][:, -1, :, :, :]  # last timestep

        pooled = F.adaptive_avg_pool2d(last_hidden, 1).squeeze(-1).squeeze(-1)
        out = self.fc(pooled)
        return out
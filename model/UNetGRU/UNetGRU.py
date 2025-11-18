import torch
import torch.nn as nn

from model.UNet.UNet import UNet


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, stride=1, N=4):
        """
        ConvGRUCell with configurable gate blocks (update, reset, candidate).
        Each gate uses a double-conv block: Conv → BN → ReLU → Conv → BN

        Args:
            input_dim (int): feature channels of one view (before N)
            hidden_dim (int): hidden state channels (before N). If None, = input_dim
            kernel_size (int): kernel for all gate convolutions
            N (int): number of input views (UNet 会先 concat N 视角，所以这里用于通道放大)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        input_dim *= N
        hidden_dim *= N

        self.hidden_dim = hidden_dim
        self.N = N

        # =============== Gate Blocks ===============
        # Update gate z_t
        self.conv_z = self.get_gate_block(
            in_ch=input_dim + hidden_dim,
            out_ch=hidden_dim,
            kernel_size=kernel_size
        )

        # Reset gate r_t
        self.conv_r = self.get_gate_block(
            in_ch=input_dim + hidden_dim,
            out_ch=hidden_dim,
            kernel_size=kernel_size
        )

        # Candidate memory h_tilde
        # 注意这里输入是: concat([x, r * h_prev])
        self.conv_h = self.get_gate_block(
            in_ch=input_dim + hidden_dim,
            out_ch=hidden_dim,
            kernel_size=kernel_size
        )

        # hidden state storage
        self.h_prev = None

    # =============== Gate Block (Double-Conv) ===============
    @staticmethod
    def get_gate_block(in_ch, out_ch, kernel_size=3):
        """
        A stronger gate block: Conv → BN → ReLU → Conv → BN
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_ch),
        )

    def reset_state(self):
        """清空隐藏状态。开始新视频序列时调用。"""
        self.h_prev = None

    # =============== Forward ===============
    def forward(self, x):
        """
        Args:
            x: B × (C*N) × H × W   （已经 concat 了 N 个视角）
        Returns:
            h_new: new hidden state with same shape as x (B × hidden_dim_total × H × W)
        """

        # 初始化 hidden state（第一次调用或 reset_state() 之后）
        if self.h_prev is None:
            self.h_prev = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3),
                device=x.device, dtype=x.dtype
            )

        # 按通道拼接 x 和 h_prev
        combined = torch.cat([x, self.h_prev], dim=1)

        # 1) Update gate z_t
        z = torch.sigmoid(self.conv_z(combined))

        # 2) Reset gate r_t
        r = torch.sigmoid(self.conv_r(combined))

        # 3) Candidate memory h~_t
        combined_candidate = torch.cat([x, r * self.h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_candidate))

        # 4) Final new hidden state
        h_new = (1 - z) * self.h_prev + z * h_tilde
        self.h_prev = h_new

        return h_new


class UNetGRU(UNet):
    """
    在原始 UNet 的 bottleneck (d7) 上插入 ConvGRU，
    用于视频时序一致性；兼容图片 & 视频两种模式。
    """

    def __init__(self):
        super().__init__()

        # 假设 UNet 中:
        #   self.c7: bottleneck 单视角通道数
        #   self.N : 视角数量（4 fish-eyes）
        # ConvGRU 以“单视角通道数”为 input_dim/hidden_dim，内部会乘 N
        self.gru_cell = ConvGRUCell(
            input_dim=self.c7,
            hidden_dim=self.c7,
            kernel_size=3,
            N=self.N
        )

        # True: 视频模式（跨 forward 保留时序状态）
        # False: 图片模式（每次 forward 都 reset GRU，但仍然经过 GRU 结构）
        self.gru_temporal_enabled = True

    # =============== GRU 模式控制接口 ===============
    def enable_GRU(self, flag: bool = True):
        """
        控制是否启用“跨 forward 的时序记忆”。

        - flag=True : 视频模式（连续帧时使用）
        - flag=False: 图片模式（每个 batch 独立，GRU 每次重置）

        注意：不管 True/False，forward 中都会经过 ConvGRU 的结构，只是
        False 时不会累积之前帧的 hidden state。
        """
        self.gru_temporal_enabled = flag

    def reset_GRU(self):
        """
        手动清空 GRU 状态。新视频序列开始时建议调用。
        """
        self.gru_cell.reset_state()

    # =============== Forward ===============
    def forward(self, inputs, return_encoding: bool = True):
        """
        Args:
            inputs: [B, N, 3, H, W]  (N 视角输入)
            return_encoding: 是否返回 bottleneck 编码特征

        Returns:
            out:         解码后的输出（通常是拼接权重/偏移/颜色等再走后级）
            encoding_map: GRU 处理后的最深层特征（可喂给 RegNet 等）
        """

        # 如果禁用时序记忆，则在每个 forward 开始时重置 hidden state
        if not self.gru_temporal_enabled:
            self.reset_GRU()

        B, N, C, H, W = inputs.shape
        assert N == self.N, f"Expected N={self.N} views, but got N={N}"

        # -------- Encode each view separately --------
        features = []
        for i in range(self.N):
            view = inputs[:, i]  # [B, 3, H, W]
            feats = self.encode_single(view)  # list of [d1..d7] for this view
            features.append(feats)

        # -------- Concatenate features of all views per encoder layer --------
        cat_features = []
        # 7 层 encoder 输出: d1..d7
        for layer in range(7):
            cat_layer = torch.cat([f[layer] for f in features], dim=1)
            cat_features.append(cat_layer)

        # -------- Decode with skip connections --------
        d1, d2, d3, d4, d5, d6, d7 = cat_features  # d7: [B, c7*N, H/64, W/64]

        # 在最深层特征上应用 ConvGRU
        d7 = self.gru_cell(d7)  # apply ConvGRU on the deepest features

        encoding_map = d7  # deepest feature (Encoding Map)

        u7 = self.up7(d7)
        u7 = self.conv7(torch.cat([u7, d6], dim=1))

        u6 = self.up6(u7)
        u6 = self.conv6(torch.cat([u6, d5], dim=1))

        u5 = self.up5(u6)
        u5 = self.conv5(torch.cat([u5, d4], dim=1))

        u4 = self.up4(u5)
        u4 = self.conv4(torch.cat([u4, d3], dim=1))

        u3 = self.up3(u4)
        u3 = self.conv3(torch.cat([u3, d2], dim=1))

        u2 = self.up2(u3)
        u2 = self.conv2(torch.cat([u2, d1], dim=1))

        u1 = self.up1(u2)
        out = self.conv1(u1)

        if return_encoding:
            return out, encoding_map
        return out


if __name__ == "__main__":
    model = UNetGRU()

    # 假设 N=4 视角
    B, N, C, H, W = 2, model.N, 3, 256, 512
    imgs = torch.randn(B, N, C, H, W)  # [B, N, 3, H, W]

    # 1) 图片模式（不跨 batch 累积时序）
    model.enable_GRU(False)
    out, enc = model(imgs, return_encoding=True)
    print("=== Image mode ===")
    print("Input Shape:        ", imgs.shape)
    print("Decoder output shape:", out.shape)
    print("Encoding map shape:  ", enc.shape)

    # 2) 视频模式（模拟连续两帧）
    model.enable_GRU(True)
    model.reset_GRU()  # 新视频开始

    frame1 = torch.randn(B, N, C, H, W)
    frame2 = torch.randn(B, N, C, H, W)

    out1, enc1 = model(frame1, return_encoding=True)
    out2, enc2 = model(frame2, return_encoding=True)
    print("=== Video mode (2 frames) ===")
    print("Frame1 out shape:", out1.shape, "enc1:", enc1.shape)
    print("Frame2 out shape:", out2.shape, "enc2:", enc2.shape)

import torch
import torch.nn as nn

from model.UNet.UNet import UNet

class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, kernel_size=3, stride=1, N=4):
        """
        ConvGRUCell with configurable gate blocks (update, reset, candidate).
        Each gate uses a double-conv block: Conv → BN → ReLU → Conv → BN
        
        Args:
            input_dim (int): feature channels of one view
            hidden_dim (int): hidden state channels (before N multiply)
            kernel_size (int): kernel for all gate convolutions
            N (int): number of input views (you concat input_dim * N)
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        # For multi-view stitching
        input_dim *= N
        hidden_dim *= N

        padding = kernel_size // 2
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
        A stronger gate block: Conv → BN → ReLU → Conv
        """
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size//2),
            nn.BatchNorm2d(out_ch),
        )


    # =============== Forward ===============
    def forward(self, x):
        """
        Args:
            x: B × (C*N) × H × W
        Returns:
            h_new: new hidden state with same shape
        """

        # 初始化 hidden state
        if self.h_prev is None:
            self.h_prev = torch.zeros(
                x.size(0), self.hidden_dim, x.size(2), x.size(3),
                device=x.device
            )

        # 按通道拼接 x 和 h_prev
        combined = torch.cat([x, self.h_prev], dim=1)

        # 1) Update gate z
        z = torch.sigmoid(self.conv_z(combined))

        # 2) Reset gate r
        r = torch.sigmoid(self.conv_r(combined))

        # 3) Candidate memory h_tilde
        combined_candidate = torch.cat([x, r * self.h_prev], dim=1)
        h_tilde = torch.tanh(self.conv_h(combined_candidate))

        # 4) Final new hidden state
        h_new = (1 - z) * self.h_prev + z * h_tilde
        self.h_prev = h_new

        return h_new


class UNetGRU(UNet):
    def __init__(self):
        super().__init__()
        self.gru_cell = ConvGRUCell(input_dim=self.c7, kernel_size=3)

    def forward(self, inputs, return_encoding=True):
        features = []
        for i in range(self.N):
            view = inputs[:, i]  # [B, 3, H, W]
            feats = self.encode_single(view)
            features.append(feats)

        # -------- Concatenate features of all views per encoder layer --------
        cat_features = []
        for layer in range(7):
            cat_layer = torch.cat([f[layer] for f in features], dim=1)
            cat_features.append(cat_layer)

        # -------- Decode with skip connections --------
        d1, d2, d3, d4, d5, d6, d7 = cat_features

        d7 = self.gru_cell(d7) # apply ConvGRU on the deepest features
        
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
    imgs = torch.randn(2, 4, 3, 256, 512)  # [B, N, 3, H, W]
    out, enc = model(imgs, return_encoding=True)
    print("Input Shape:", imgs.shape)
    print("Decoder output shape:", out.shape)
    print("Encoding map shape:", enc.shape)


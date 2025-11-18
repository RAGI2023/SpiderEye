import torch
import torch.nn as nn
import torch.nn.functional as F


# =================== Basic building blocks ===================
class Upsize(nn.Module):
    """Upsampling module using bilinear interpolation."""
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        _, _, h, w = x.shape
        nh, nw = h * self.scale_factor, w * self.scale_factor
        return F.interpolate(x, [nh, nw], mode=self.mode, align_corners=self.align_corners)


def downconv_double(in_dim, out_dim, kernel):
    """Two convolutional layers with downsampling (stride=2)."""
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ELU(),
        nn.Conv2d(out_dim, out_dim, kernel, stride=2, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_dim),
        nn.ELU()
    )


def upconv_single(in_dim, out_dim, kernel):
    """Upsampling followed by one convolution."""
    return nn.Sequential(
        Upsize(2, 'bilinear'),
        nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ELU()
    )


def identity_conv(in_dim, out_dim):
    """Conv layer used after skip concatenation to reduce feature dimension."""
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_dim),
        nn.ELU()
    )


# =================== Full LargeUNet ===================
class UNet(nn.Module):
    """
    Reconstructed version of the U-Net used in:
    "Weakly-Supervised Stitching Network for Real-World Panoramic Image Generation"
    
    Features:
    - Independent encoding for each input image
    - Multi-view feature concatenation at each encoder stage
    - Standard U-Net decoder with skip connections
    - Option to return the Encoding Map (deep feature representation)
    """
    def __init__(self):
        super().__init__()
        # Channel configuration (same as in the paper)
        c1, c2, c3, c4, c5, c6, c7, cf = 16, 32, 64, 128, 256, 256, 512, 32
        self.N = 4  # Number of input views
        self.c7 = c7 # for GRU input dimension
        # ------- Encoder -------
        self.down1 = downconv_double(3, c1, 7)
        self.down2 = downconv_double(c1, c2, 5)
        self.down3 = downconv_double(c2, c3, 3)
        self.down4 = downconv_double(c3, c4, 3)
        self.down5 = downconv_double(c4, c5, 3)
        self.down6 = downconv_double(c5, c6, 3)
        self.down7 = downconv_double(c6, c7, 3)

        # ------- Decoder -------
        self.up7 = upconv_single(c7 * self.N, c6 * self.N, 3)
        self.up6 = upconv_single(c6 * self.N, c5 * self.N, 3)
        self.up5 = upconv_single(c5 * self.N, c4 * self.N, 3)
        self.up4 = upconv_single(c4 * self.N, c3 * self.N, 3)
        self.up3 = upconv_single(c3 * self.N, c2 * self.N, 3)
        self.up2 = upconv_single(c2 * self.N, c1 * self.N, 3)
        self.up1 = upconv_single(c1 * self.N, cf, 3)

        # ------- Skip connection fusion layers -------
        self.conv7 = identity_conv(c6 * self.N * 2, c6 * self.N)
        self.conv6 = identity_conv(c5 * self.N * 2, c5 * self.N)
        self.conv5 = identity_conv(c4 * self.N * 2, c4 * self.N)
        self.conv4 = identity_conv(c3 * self.N * 2, c3 * self.N)
        self.conv3 = identity_conv(c2 * self.N * 2, c2 * self.N)
        self.conv2 = identity_conv(c1 * self.N * 2, c1 * self.N)
        self.conv1 = identity_conv(cf, 16)

    # -------------------------------------------------
    def encode_single(self, x):
        """Encode one input image and return all intermediate feature maps."""
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        return [d1, d2, d3, d4, d5, d6, d7]

    # -------------------------------------------------
    def forward(self, inputs, return_encoding=True):
        """
        Forward pass of UNet.

        Args:
            inputs: Tensor of shape [B, N, 3, H, W]
            return_encoding (bool): if True, return both decoder output and encoding map

        Returns:
            out: decoder output (high-resolution feature map)
            encoding_map (optional): deep encoded multi-view representation
        """

        # -------- Encode each view independently --------
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

# =================== Quick test ===================
if __name__ == "__main__":
    model = UNet()
    imgs = torch.randn(2, 4, 3, 256, 512)  # [B, N, 3, H, W]
    out, enc = model(imgs, return_encoding=True)
    print("Decoder output shape:", out.shape)
    print("Encoding map shape:", enc.shape)

"""
Concise reimplementation of the Homography/Affine regressor used in
Weakly-Supervised Stitching Network.

Interface and network topology are kept identical to the original:
- Base class: Regressor
- Concrete class: LargeRegressor

Forward contract:
- Input: feature tensor, typically [B, 3*512, 2, 4]
- Output: theta of shape [B, homography*3, 2, 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Regressor(nn.Module):
    """
    Homography/Affine estimation head.
    Keeps the same behavior as networks/homography/base.py
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.homography = opt.homography
        # Fixed to 4 as in the original implementation
        self.input_direction = 4
        # Match original behavior: when using 'large' UNet, expect 4*512 input channels
        if getattr(opt, 'unet', 'large').lower() == 'large':
            self.first_layer = self.input_direction * 512  # 4 * 512 = 2048

    @staticmethod
    def conv_block(in_dim, out_dim, kernel_size=3, stride=1):
        """Conv2d + BatchNorm + ELU (padding fixed to 1 as original)."""
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=1),
            nn.BatchNorm2d(out_dim, momentum=0.1),
            nn.ELU(),
        )

    def _forward(self, feat):
        """
        Internal forward: from feature map to flattened theta vector.
        Expects conv layers and fc to be defined in subclass.
        """
        x = feat
        x = self.conv45(x)
        x = self.conv5a(x)
        x = self.conv5b(x)
        x = self.conv56(x)
        x = self.conv6a(x)
        x = self.conv6b(x)
        # Global average pool to [B, 512]
        x = F.avg_pool2d(x, (x.shape[2], x.shape[3]))
        x = x.view(-1, x.shape[1])
        theta = self.fc(x)  # [B, 6 * homography * input_direction]
        return theta

    def forward(self, feat):
        """Return theta shaped as [B, homography*input_direction, 2, 3]."""
        theta = self._forward(feat)
        theta = theta.view(-1, self.homography * 4, 2, 3) # 2*3 affine
        return theta


class LargeRegressor(Regressor):
    """Identical architecture to the original LargeRegressor."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Conv stages
        self.conv45 = self.conv_block(self.first_layer, 512, kernel_size=3, stride=2)
        self.conv5a = self.conv_block(512, 512, kernel_size=3, stride=1)
        self.conv5b = self.conv_block(512, 512, kernel_size=3, stride=1)
        self.conv56 = self.conv_block(512, 512, kernel_size=1, stride=2)
        self.conv6a = self.conv_block(512, 512, kernel_size=1, stride=1)
        self.conv6b = self.conv_block(512, 512, kernel_size=1, stride=1)

        # FC head with identity-affine bias initialization
        out_dim = 6 * self.homography * self.input_direction
        self.fc = nn.Linear(512, out_dim)
        with torch.no_grad():
            self.fc.weight.zero_()
            bias = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float32).repeat(self.homography * self.input_direction)
            self.fc.bias.copy_(bias)


def build_regressor(opt):
    """Factory to match original builder semantics (only 'large' supported)."""
    call = getattr(opt, 'reg', 'large')
    if str(call).lower() == 'large':
        return LargeRegressor(opt)
    raise ValueError(f'Not supported regressor: {call}')




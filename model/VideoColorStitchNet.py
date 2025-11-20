import torch
import torch.nn as nn

from model.ColorStitchNet import ColorStitchNet

class VideoColorStitchNet(ColorStitchNet):
    def __init__(self, opt, device):
        super().__init__(opt, device)
        

    def reset_state(self):
        """reset GRU (if exists) for new clip"""
        if hasattr(self.backbone, "reset_GRU"):
            self.backbone.reset_GRU()

    def forward_clip(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clip: [B, T, N, 3, H, W]
        Returns:
            panoramas: [B, T, 3, H, W]
        """
        B, T, N, C, H, W = clip.shape

        self.reset_state()
        self.recorded_flow = [] # for loss computation

        pano_list = []

        for t in range(T):
            images_t = clip[:, t, ...]         # [B, N, 3, H, W]

            panorama_t = super().forward(images_t)   # [B, 3, H, W]
            self.recorded_flow.append(self.flow_map) # for loss computation
            pano_list.append(panorama_t)

        self.recorded_flow = torch.stack(self.recorded_flow, dim=1)  # [B, T, N, H, W, 2 * homography]

        return torch.stack(pano_list, dim=1)          # [B, T, 3, H, W]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, N, 3, H, W]
        Returns:
            panoramas: [B, T, 3, H, W]
        """
        return self.forward_clip(x)
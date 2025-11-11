import torch
import torch.nn as nn

from model.StitchNet import HomoDispNet

class ColorStitchNet(HomoDispNet):
    """Color StitchNet"""
    def __init__(self, opt, device):
        super(ColorStitchNet, self).__init__(opt, device)
        self.correct_block = self.get_correct_block(in_dim=16, out_dim=3 * self.input_direction)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, N, C, H, W]  # N = input_direction (views)
        Returns:
            panorama: [B, C, H, W]
        """
        B, N, C, H, W = images.shape
        assert C == 3, f"expected 3 channels, got {C}"
        # 将 input_direction 与实际 N 对齐
        self.input_direction = N
        
        iconv_1, downfeature = self.backbone(images)    # iconv_1: [B, 16, H, W]

        if self.use_kl:
            # KL latent sampling
            mu = self.mu_head(downfeature)
            logvar = self.logvar_head(downfeature)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            theta  = self.Regressor(z)        # [B, homography*N, 2, 3]
        else:
            theta  = self.Regressor(downfeature)        # [B, homography*N, 2, 3]
        
        self.theta = theta  # for loss computation
        self.flow_map = []

        # 生成每个方向/每个单应的权重 & 位姿参数
        weight = self.weight_block(iconv_1)         # [B, homography*N, H, W]
        if self.record_weights:
            self.weights = weight.detach().cpu()


        # 局部位移（disp） -> [B, H, W, 2*homography*N]
        disp = self.local_limit * self.local_adj_block(iconv_1)  
        disp = disp.permute(0, 2, 3, 1).contiguous() # [B, 2*h*N, H, W]
        pre_correct_map = self.correct_block(iconv_1)  # [B, 3*N, H, W]
        # 输出全景
        panorama = torch.zeros([B, C, H, W], device=self.device, dtype=images.dtype)
        self.warped = []  # for recording warped images if needed
        for i in range(N):
            # 取第 i 个视角的图像 [B, 3, H, W]
            img_i = images[:, i, ...]
            start, end = i * self.homography, (i + 1) * self.homography

            # 估计 flow（注意：在 HomoDispNet.flow_estimation 里已经把 disp 的该方向切片加进去了）
            flow = self.flow_estimation(
                i,                      # direction
                disp,                   # for picking adj slice inside
                theta[:, start:end, :, :],  # thetas for this direction
                imgsize=[H, W],
            )  # -> [B, H, W, 2*homography]

            self.flow_map.append(flow) # for loss computation
            # 进行颜色校正
            img_i = self.correct(img_i, pre_correct_map[:, i*3: i*3+3])
            # 对该方向的每个单应 warp（返回 [Homography, B, C, H, W]）
            warped_images = self.warp(flow, img_i)
            if self.record_warped:
                self.warped.append(warped_images[0][0].detach().cpu()) # [C, H, W]
            # 以该方向的权重进行融合
            weight_i = weight[:, start:end, ...]     # [B, homography, H, W]
            panorama += self.weighted_sum(warped_images, weight_i)

        if self.use_kl:
            return panorama, mu, logvar
        else:
            return panorama, None, None
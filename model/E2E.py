import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import math

from model.Unet.Unet import UNet
from model.Regnet.Regnet import Regressor

# ==========================================================
# E2E (includes BaseNet logic)
# ==========================================================
class MetaStitcher(nn.Module):
    """
    Base Stitcher Network
    Provides core utilities for multi-homography image stitching
    """
    def __init__(self, opt, device):
        super().__init__()
        self.opt = opt
        self.device = device

        # --- Basic network configuration ---
        self.input_direction = 4
        self.homography = opt.homography

        # --- Submodules ---
        self.UNet = UNet()
        self.Regressor = Regressor(opt)
        self.weight_block = self.get_weight_block(16)
        self.normalizer = ImageNormalize(opt.mean, opt.std)

        # --- Initialize weights if specified ---
        if hasattr(opt, "init_model"):
            self.init_weights(opt)

    # ==========================================================
    # Weight initialization 
    # ==========================================================
    def init_weights(self, opt):
        init_type = opt.init_model
        gain = opt.init_gain

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and ('Conv' in classname or 'Linear' in classname):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=math.sqrt(5), mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif 'BatchNorm2d' in classname:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    # ==========================================================
    # Core blocks
    # ==========================================================
    def get_weight_block(self, in_dim):
        """Generate weight map for homography blending"""
        return nn.Sequential(
            nn.Conv2d(in_dim, self.homography * self.input_direction, 3, 1, 1),
            nn.Softmax(dim=1)
        )

    def get_displace_block(self, in_dim):
        """Local flow adjustment block"""
        return nn.Sequential(
            nn.Conv2d(in_dim, 2 * self.homography * self.input_direction, 3, 1, 1),
            nn.Tanh()
        )

    @staticmethod
    def get_correct_block(in_dim, out_dim):
        """Pixel-wise color correction block"""
        return nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim, momentum=0.1),
            nn.ELU(),
            nn.Conv2d(in_dim, out_dim, 3, 1, 1),
            nn.BatchNorm2d(out_dim, momentum=0.1),
            nn.ELU(),
            nn.Conv2d(out_dim, out_dim, 3, 1, 1),
            nn.Tanh()
        )

    # ==========================================================
    # Core functional utilities
    # ==========================================================
    @staticmethod
    def flow_estimation(theta: torch.Tensor, imgsize):
        """Compute flow maps from affine matrices"""
        flow_list = []
        h, w = imgsize
        n, c, *_ = theta.shape
        for i in range(c):
            flow = F.affine_grid(theta[:, i, :, :], size=[n, c, h, w], align_corners=True)
            flow_list.append(flow)
        return torch.cat(flow_list, dim=3)  # [B, H, W, 2*Homography]

    @staticmethod
    def warp(flow_map, image) -> torch.Tensor:
        """Apply warping given flow maps"""
        warped_list = []
        *_, h = flow_map.shape
        homo_num = h // 2
        for i in range(homo_num):
            warped = F.grid_sample(image, flow_map[..., 2*i:2*i+2],
                                   padding_mode='border', align_corners=True)
            warped_list.append(warped)
        return torch.stack(warped_list)  # [Homography, B, C, H, W]

    def weighted_sum(self, images, weights):
        """Blend warped images according to weights
        images: [Homography, B, C, H, W]
        weights: [B, homography, H, W]
        """
        _, *size = images.shape
        output = torch.zeros(size).to(self.device)
        for i, img in enumerate(images):
            output += img * weights[:, [i], ...].repeat(1, 3, 1, 1)
        return output

    def correct(self, img, color_map):
        """Color correction"""
        img = img + color_map * img * (1 - img)
        img = self.normalizer(img)
        return img


# ==========================================================
# Helper normalization modules
# ==========================================================
class ImageNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.function = transforms.Normalize(mean=mean, std=std, inplace=False)

    def forward(self, x):
        return self.function(x)


class ImageDenormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        mean = [-m / s for m, s in zip(mean, std)]
        std = [1 / s for s in std]
        self.function = transforms.Normalize(mean=mean, std=std, inplace=False)

    def forward(self, x):
        return self.function(x)

class HomoDispNet(MetaStitcher):
    def __init__(self, opt, device):
        super().__init__(opt, device)
        self.local_limit = self.opt.local_adj_limit  # if 0, No Local Adjustment
        self.local_adj_block = self.get_displace_block(16)
        self.record_weights = opt.get('record_weights', False)
        self.record_warped = opt.get('record_warped', False)
        self.warped = None
        self.weights = None
        if opt.get('kl', False):
            self.use_kl = True
            self.mu_head = nn.Conv2d(2048, 2048, 1)
            self.logvar_head = nn.Conv2d(2048, 2048, 1)
        else:
            self.use_kl = False
        
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

        # UNet 已改为支持 [B, N, 3, H, W]
        iconv_1, downfeature = self.UNet(images)    # iconv_1: [B, 16, H, W]

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

        # 生成每个方向/每个单应的权重 & 位姿参数
        weight = self.weight_block(iconv_1)         # [B, homography*N, H, W]
        if self.record_weights:
            self.weights = weight.detach().cpu()


        # 局部位移（disp） -> [B, H, W, 2*homography*N]
        disp = self.local_limit * self.local_adj_block(iconv_1)  
        disp = disp.permute(0, 2, 3, 1).contiguous() # [B, 2*h*N, H, W]

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

            # 对该方向的每个单应 warp（返回 [Homography, B, C, H, W]）
            warped_images = self.warp(flow, img_i)
            if self.record_warped:
                self.warped.append(warped_images[0][0].detach().cpu()) # [C, H, W]]
            # 以该方向的权重进行融合
            weight_i = weight[:, start:end, ...]     # [B, homography, H, W]
            panorama += self.weighted_sum(warped_images, weight_i)

        if self.use_kl:
            return panorama, mu, logvar
        else:
            return panorama, None, None


    def flow_estimation(self, direction: int, disp: torch.Tensor, *args, **kwargs):
        flow = super().flow_estimation(*args, **kwargs)
        start, end = direction * self.homography, direction * self.homography + self.homography
        adj = disp[..., start * 2: end * 2]

        final_flow = flow + adj

        return final_flow


if __name__ == "__main__":
    from argparse import Namespace
    opt = Namespace()
    opt.homography = 2
    opt.local_adj_limit = 0.05
    opt.mean = [0.485, 0.456, 0.406]
    opt.std = [0.229, 0.224, 0.225]
    # opt.init_model = 'large'
    opt.init_gain = 0.02

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HomoDispNet(opt, device).to(device)
    # print(model)

    x = torch.randn(4, 2, 3, 256, 256).to(device)  # [input_direction x B x C x H x W]
    with torch.no_grad():
        y = model(x)
    print(y.shape)
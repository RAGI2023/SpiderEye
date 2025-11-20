import torch
from model.loss.common_loss import create_window, _ssim
from model.loss.vgg_loss import VGGPerceptualLoss

def l_num_loss_time(pred: torch.Tensor, target: torch.Tensor, num: int = 1) -> torch.Tensor:
    """
    Generalized L_num loss for regression tasks.

    Args:
        pred (torch.Tensor): Predicted values.
        target (torch.Tensor): Ground truth values.
        num (int): The power to which the absolute difference is raised. Default is 1.

    Returns:
        torch.Tensor: Computed L_num loss.
    """
    diff = torch.abs(pred - target)
    diff = torch.mean(diff, dim=1, keepdim=False)
    loss = diff.pow(num)
    return loss.mean()

def ssim_loss_time(img1, img2, window_size=11, size_average=True, is_train=False):
    """
    img1, img2: [B, T, C, H, W]
    returns: scalar loss averaged over T
    """
    B, T, C, H, W = img1.shape

    # Flatten time dimension → (B*T, C, H, W)
    img1_f = img1.reshape(B*T, C, H, W)
    img2_f = img2.reshape(B*T, C, H, W)

    # 创建 window
    window = create_window(window_size, C).to(img1.device).type_as(img1)

    # 逐帧 SSIM
    ssim_map = _ssim(img1_f, img2_f, window, window_size, C, size_average=False)  # shape: [B*T]

    # reshape 回 T，将每个 Batch 的 T 帧平均
    ssim_map = ssim_map.reshape(B, T)

    # 时间维平均
    ssim_t = ssim_map.mean(dim=1)  # shape [B]

    # size_average → 对 batch 取平均
    ssim_final = ssim_t.mean() if size_average else ssim_t

    if is_train:
        return 1. - torch.clamp(ssim_final, 0., 1.)
    else:
        return ssim_final

class VggPerceptualLossTime(VGGPerceptualLoss):
    def __init__(self):
        super().__init__(layers=(2, 3, 4), weights=(1.0, 0.5, 0.25))
    
    def forward(self, pred, target):
        """
        pred, target: [B, T, C, H, W]
        returns: scalar loss averaged over T
        """
        B, T, C, H, W = pred.shape

        # Flatten time dimension → (B*T, C, H, W)
        pred_f = pred.reshape(B*T, C, H, W)
        target_f = target.reshape(B*T, C, H, W)

        # 计算 VGG 感知损失
        loss = super().forward(pred_f, target_f)  # scalar

        return loss
# def flow_continous_loss(flow):
    """
    Enforce temporal continuity in predicted optical flow fields.
    
    Args:
        flow (torch.Tensor): Predicted flow fields of shape (B, T, 2, H, W).
    """



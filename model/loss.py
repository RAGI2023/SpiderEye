import torch
import torch.nn.functional as F
import math

def l_num_loss(pred: torch.Tensor, target: torch.Tensor, num: int = 1) -> torch.Tensor:
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
    loss = diff.pow(num)
    return loss.mean()

def gaussian(window_size, sigma):
    """Gaussian Blur"""
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    """Create Window"""
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    """Calculate SSIM Score"""
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    """SSIM Loss Class"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        """Process"""
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim_loss(img1, img2, window_size=11, size_average=True, is_train=False):
    """
    Range Assumption: 0 ~ 1
    The engine has been integrated with this function. (2021-03-14)
    :param img1: mini-batch of ground-truth image [Batch X Channel X Height X Width]
    :param img2: mini-batch of predicted image [Batch X Channel X Height X Width]
    :param window_size:
    :param size_average:
    :param is_train: set "True" if you train the model.
    :return:
    """
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    if is_train:
        return 1. - torch.clamp(_ssim(img1, img2, window, window_size, channel, size_average), min=0., max=1.)
    else:
        return _ssim(img1, img2, window, window_size, channel, size_average)

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self, eps=1e-6):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = eps

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def gradient_loss(pred, target):
    """Compute gradient-based loss (edge consistency)."""
    # 1. 定义 Sobel 核
    sobel_x = torch.tensor([[[[-1, 0, 1],
                              [-2, 0, 2],
                              [-1, 0, 1]]]], dtype=torch.float32)
    sobel_y = torch.tensor([[[[-1, -2, -1],
                              [ 0,  0,  0],
                              [ 1,  2,  1]]]], dtype=torch.float32)

    # 2. 复制到所有通道
    C = pred.shape[1]
    sobel_x = sobel_x.repeat(C, 1, 1, 1).to(pred.device)
    sobel_y = sobel_y.repeat(C, 1, 1, 1).to(pred.device)

    # 3. 计算梯度
    gx_pred = F.conv2d(pred, sobel_x, padding=1, groups=C)
    gy_pred = F.conv2d(pred, sobel_y, padding=1, groups=C)
    gx_tgt  = F.conv2d(target, sobel_x, padding=1, groups=C)
    gy_tgt  = F.conv2d(target, sobel_y, padding=1, groups=C)

    # 4. 用 Charbonnier 或 L1 计算差异
    diff_x = torch.abs(gx_pred - gx_tgt)
    diff_y = torch.abs(gy_pred - gy_tgt)
    grad_loss = (diff_x + diff_y).mean()

    return grad_loss


def affine_loss(theta):
    # 仿射部分
    I = torch.tensor([[1, 0, 0],
                      [0, 1, 0]], dtype=torch.float32, device=theta.device)
    I = I.view(1, 1, 2, 3).expand_as(theta)
    loss_affine = F.mse_loss(theta, I)
    return loss_affine



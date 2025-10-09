import random
import torch
import os
import datetime as dt

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # 更快
    torch.backends.cudnn.benchmark = True       # 根据输入尺寸自动选择最快算法

def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def format_secs(s: float) -> str:
    return str(dt.timedelta(seconds=int(s)))


def save_ckpt(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

import torch

import torch

def stitch2img(img_front, img_left, learned_mask)->torch.Tensor:
    """
    拼接两个输入图像（前视图、左视图）生成画布级别的 stitched 图像。
    适用于 learned_mask 为画布宽度（即网络输入 front_canvas / left_canvas 时）的情况。

    参数:
        img_front: (B,3,H,canvas_w)           前视图
        img_left:  (B,3,H,canvas_w)           左视图
        learned_mask: (B, 1, H, canvas_w)  网络预测的融合权重 (Wc = W + W - overlap_px)
        overlap_ratio: float,             重叠比例 (默认 0.25)

    返回:
        stitched_img: (B, 3, H, Wc)       拼接画布
    """
    # 初始化画布
    stitched_img = torch.zeros_like(img_front).to(img_front.device).float()

    stitched_img = img_left * learned_mask + img_front * (1 - learned_mask)

    return stitched_img




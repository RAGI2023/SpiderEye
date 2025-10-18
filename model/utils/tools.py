import random
import torch
import os
import datetime as dt
import torch.nn as nn

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


def convert_bn_to_gn(module, max_groups=32):
    """Recursively replace all BatchNorm2d with GroupNorm intelligently."""
    for name, m in module.named_children():
        if isinstance(m, nn.BatchNorm2d):
            c = m.num_features
            # 自动找出能整除的最优 group 数
            g = min(max_groups, c)
            while c % g != 0 and g > 1:
                g -= 1
            # 如果通道太小（如3），跳过或转 InstanceNorm
            if c < 8:
                print(f"[convert_bn_to_gn] Small channel ({c}), using InstanceNorm2d.")
                new_m = nn.InstanceNorm2d(c, affine=True)
            else:
                new_m = nn.GroupNorm(g, c)
            setattr(module, name, new_m)
        else:
            convert_bn_to_gn(m, max_groups)
    return module

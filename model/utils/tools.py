import random
import torch
import os
import datetime as dt
import torch.nn as nn
import torch.distributed as dist
from glob import glob
import argparse

def check_cfg_keys(g_cfg):
    g_cfg.train.lr = float(g_cfg.train.lr)
    g_cfg.train.batch_size = int(g_cfg.train.batch_size)
    g_cfg.train.epochs = int(g_cfg.train.epochs)
    g_cfg.train.num_workers = int(g_cfg.train.num_workers)
    g_cfg.train.weight_decay = float(g_cfg.train.weight_decay)
    g_cfg.model.mean = tuple(map(float, g_cfg.model.mean.split(',')))
    g_cfg.model.std = tuple(map(float, g_cfg.model.std.split(',')))
    g_cfg.model.type = g_cfg.model.get('type', 'UNet')
    g_cfg.train.eval = True


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

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_train', action='store_true', help="Continue training from latest checkpoint")
    return parser.parse_args()

def get_latest_ckpt(ckpt_dir):
    ckpts = [
        f for f in glob(os.path.join(ckpt_dir, "*.pth"))
        # if not os.path.basename(f).startswith("best")
    ]
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts[-1] if ckpts else None
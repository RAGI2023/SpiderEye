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

def ensure_tensor_imgs(batch):
    """兼容 Dataset 返回 imgs 或 (imgs, path) 的情况；并做基本维度断言"""
    if isinstance(batch, (list, tuple)):
        imgs = batch[0]
    else:
        imgs = batch
    assert imgs.dim() == 5, f"Expect imgs of shape [B, 6, C, H, W], got {tuple(imgs.shape)}"
    return imgs

def save_ckpt(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
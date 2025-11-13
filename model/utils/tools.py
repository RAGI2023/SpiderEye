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
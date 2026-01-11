import yaml, time, os
import argparse
from easydict import EasyDict as edic
import random
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import cv2
import numpy as np

from model.utils.EquiDataset import EquiDataset
from model.utils.FishEyeDataset import FishEyeDataset
from model.utils.EquiVideoDataset import EquiVideoDataset

from model.ColorStitchNet import ColorStitchNet
from model.VideoColorStitchNet import VideoColorStitchNet


# =======================================================
# 工具：把 fisheye [4, 3, Hc, Wc] 拼到一个 2×2 四宫格 (与 EquiDataset 同布局)
# =======================================================
def make_fisheye_grid(fisheye_tensor):
    """
    fisheye_tensor: [4,3,Hc,Wc]
    return:         Hc×Wc×3 BGR uint8
    """
    views = fisheye_tensor  # [4,3,H,W]
    _, _, H, W = views.shape

    canvas = np.zeros((H, W, 3), dtype=np.uint8)

    # Layout 与 EquiVideoDataset._place_on_canvas 完全一致
    # view_idx: 0=front,1=right,2=back,3=left

    # front → 左上
    canvas[:, :W // 2, :] = (views[2].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # right → 中间
    interval = W // 4
    canvas[:, interval + 1: interval + 1 + W // 2, :] = \
        (views[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # back → 右侧
    canvas[:, 2 * interval: 2 * interval + 1 + W // 2, :] = \
        (views[3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    # left → wrap 在最右侧和最左侧
    left_full = (views[1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    left_w = W - 3 * interval

    canvas[:, 3 * interval:, :] = left_full[:, :left_w, :]
    canvas[:, :interval, :] = left_full[:, -interval:, :]

    return canvas  # BGR uint8


# =======================================================
# Eval 主程序
# =======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='configs/eval.yaml')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    parser.add_argument('-r', '--save_ratio', type=float, default=0.05)
    parser.add_argument('-o', '--output_dir', type=str, default='./runs/eval_outputs')
    parser.add_argument('-l', '--load_ckpt', type=int, default=None)
    args = parser.parse_args()

    # --------------------- Load config ---------------------
    with open(args.cfg, 'r') as f:
        cfg = edic(yaml.safe_load(f))

    # ======================================================
    # Dataset
    # ======================================================
    K = (0.35, -0.0015, 0.002, -0.002)
    if cfg.data.dataset_type == 'panorama':
        dataset = EquiDataset(
            folder_path=cfg.data.train_dataset,
            fov=cfg.data.fov,
            canvas_size=tuple(cfg.data.canvas_size),
            out_w=cfg.data.canvas_size[1],
            out_h=cfg.data.canvas_size[1],
            jitter_cfg={
                "random_seed": cfg.data.jitter.random_seed,
                "rotation_jitter": {
                    "yaw": cfg.data.jitter.yaw,
                    "pitch": cfg.data.jitter.pitch,
                    "roll": cfg.data.jitter.roll
                },
                "translate_range": cfg.data.jitter.translate,
                "lighting": {
                    "brightness": cfg.data.jitter.brightness,
                    "contrast": cfg.data.jitter.contrast,
                    "color_jitter": cfg.data.jitter.color_jitter
                },
                "k_jitter": cfg.data.jitter.k_jitter,
            },
            k=K,
        )

    elif cfg.data.dataset_type == 'fisheye':
        dataset = FishEyeDataset(
            folder_path=cfg.data.train_dataset,
            canvas_size=tuple(cfg.data.canvas_size),
            gt_type=cfg.data.get('gt_type', 'Samsung')
        )

    elif cfg.data.dataset_type == 'equi_video':
        dataset = EquiVideoDataset(
            video_root=cfg.data.train_dataset,
            fov=cfg.data.fov,
            canvas_size=tuple(cfg.data.canvas_size),
            out_w=cfg.data.canvas_size[1],
            out_h=cfg.data.canvas_size[1],
            overlap=cfg.data.overlap,
            clip_len=cfg.train.clip_len,
            stride=cfg.train.get('stride', 1),
            jitter_cfg=cfg.data.get("jitter", None),
            k=K,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.data.dataset_type}")

    # 强制要求 batch=1，避免视频错乱
    assert cfg.train.batch_size == 1, "❌ Eval video saving requires batch_size=1"

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        drop_last=False,
    )

    # ======================================================
    # Model
    # ======================================================
    if cfg.model.type == 'UNetGRU':
        net = VideoColorStitchNet(opt=cfg.model, device=args.device)
    else:
        net = ColorStitchNet(opt=cfg.model, device=args.device)

    net.to(args.device)
    net.eval()

    # ======================================================
    # Load checkpoint
    # ======================================================
    ckpt_dir = os.path.join('runs', cfg.experiment.load_name, 'ckpts')
    ckpt_path = (
        os.path.join(ckpt_dir, f"step_{args.load_ckpt}.pth")
        if args.load_ckpt is not None else
        os.path.join(ckpt_dir, "best.pth")
    )

    print(f"🔄 Loading checkpoint: {ckpt_path}")
    torch.serialization.add_safe_globals([edic])
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    print(f"✅ Loaded model (epoch {ckpt.get('epoch',0)}, step {ckpt.get('global_step',0)})")

    # ======================================================
    # VideoWriter 初始化
    # ======================================================
    outdir = os.path.join(cfg.experiment.name, args.output_dir)
    os.makedirs(outdir, exist_ok=True)
    Hc, Wc = cfg.data.canvas_size[1], cfg.data.canvas_size[0]
    fps = cfg.data.get("fps", 30)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    panorama_video = cv2.VideoWriter(
        os.path.join(outdir, "output_panorama.mp4"),
        fourcc, fps, (Wc, Hc)
    )
    fisheye_video = cv2.VideoWriter(
        os.path.join(outdir, "input_fisheye.mp4"),
        fourcc, fps, (Wc, Hc)
    )

    # ======================================================
    # Inference (Only save panorama output video)
    # ======================================================
    total_time = 0.0
    pbar = tqdm(total=len(loader), desc="Evaluating", ncols=120)

    for i, (fisheye_clip, gt_clip) in enumerate(loader):
        # fisheye_clip: [1,T,4,3,Hc,Wc]
        # gt_clip:      [1,T,3,Hc,Wc]

        fisheye_clip = fisheye_clip.to(args.device)

        t0 = time.perf_counter()
        outs = net(fisheye_clip)  # [1,T,3,Hc,Wc]
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        outs = outs.detach().cpu().clamp(0, 1)

        T = outs.shape[1]

        for t in range(T):
            # ---- 保存 Panorama 输出帧 ----
            frame = (outs[0, t].permute(1, 2, 0).numpy() * 255).astype('uint8')
            panorama_video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # ---- 随机保存一些图像 PNG ----
            if random.random() < args.save_ratio:
                cv2.imwrite(
                    os.path.join(args.output_dir, f"sample_{i:04d}_{t:03d}_output.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                )

        pbar.update(1)

    pbar.close()

    # 关闭视频写入器
    panorama_video.release()
    fisheye_video.release()

    avg_time = total_time / len(loader)
    print(f"🎥 Saved video to {args.output_dir}")
    print(f"⏱ Avg inference time / clip: {avg_time:.4f}s , / frame: {avg_time / cfg.data.clip_len * 1000:.2f} ms")


if __name__ == "__main__":
    with torch.no_grad():
        main()

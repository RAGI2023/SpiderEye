import yaml, time, os
import argparse
from easydict import EasyDict as edic
import random
from tqdm import tqdm

import torch
from model.utils.EquiDataset import EquiDataset
from model.utils.FishEyeDataset import FishEyeDataset
from model.ColorStitchNet import ColorStitchNet
from model.LinkNet.LinkNet import LinkNet

import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='configs/eval.yaml', help='path to config file')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='compute device')
    parser.add_argument('-r', '--save_ratio', type=float, default=0.05, help='ratio for saving images')
    parser.add_argument('-o', '--output_dir', type=str, default='./runs/eval_outputs', help='output directory')
    parser.add_argument('-l', '--load_ckpt', type=int, default=None, help='step to load model checkpoint, default best')
    args = parser.parse_args()

    # --- Load config
    with open(args.cfg, 'r') as f:
        cfg = edic(yaml.safe_load(f))

    # --- Dataset
    if cfg.data.dataset_type == 'panorama':
        dataset = EquiDataset(
            folder_path=cfg.data.train_dataset,
            fov=cfg.data.fov,
            canvas_size=(cfg.data.canvas_size[0], cfg.data.canvas_size[1]),
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
            k=(0.01, -0.1, 0.1, -0.0),
        )
    elif cfg.data.dataset_type == 'fisheye':
        dataset = FishEyeDataset(
            folder_path=cfg.data.train_dataset,
            canvas_size=(cfg.data.canvas_size[0], cfg.data.canvas_size[1]),
            gt_type=cfg.data.get('gt_type', 'Samsung')
        )
    else:
        raise ValueError(f"Unknown dataset_type: {cfg.data.dataset_type}")
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )

    # --- Model
    net = ColorStitchNet(opt=cfg.model, device=args.device)
    
    net.to(args.device)
    net.eval()

    # --- Load checkpoint
    ckpt_dir = os.path.join('runs', cfg.experiment.load_name, 'ckpts')
    ckpt_path = (
        os.path.join(ckpt_dir, f'step_{args.load_ckpt:d}.pth')
        if args.load_ckpt is not None
        else os.path.join(ckpt_dir, 'best.pth')
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"🔄 Loading checkpoint: {ckpt_path}")

    torch.serialization.add_safe_globals([edic])
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    net.load_state_dict(ckpt['model'])
    print(f"✅ Loaded from {ckpt_path} (epoch {ckpt.get('epoch', 0)}, step {ckpt.get('global_step', 0)})")

    # --- Inference
    args.output_dir = os.path.join(args.output_dir, cfg.experiment.load_name, f'epoch{ckpt.get("epoch", 0)}_step_{ckpt.get("global_step", 0)}')
    os.makedirs(args.output_dir, exist_ok=True)
    total_time = 0.0
    pbar = tqdm(total=len(loader), desc="Evaluating", ncols=100)

    for i, (imgs, gt) in enumerate(loader):
        imgs = imgs.to(args.device)

        t0 = time.perf_counter()
        outs = net(imgs)
        t1 = time.perf_counter()
        total_time += (t1 - t0)

        outs = outs.detach().cpu().clamp(0, 1)
        for b in range(outs.shape[0]):
            if random.random() < args.save_ratio:
                out_img = outs[b]
                out_img_np = (out_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                out_path = os.path.join(args.output_dir, f'sample_{i*cfg.train.batch_size + b:05d}_output.png')
                cv2.imwrite(out_path, cv2.cvtColor(out_img_np, cv2.COLOR_RGB2BGR))

                if gt is not None:
                    gt_img = gt[b].cpu()
                    gt_img_np = (gt_img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                    gt_path = os.path.join(args.output_dir, f'sample_{i*cfg.train.batch_size + b:05d}_gt.png')
                    cv2.imwrite(gt_path, cv2.cvtColor(gt_img_np, cv2.COLOR_RGB2BGR))
        pbar.update(1)

    pbar.close()
    avg_time = total_time / len(loader)
    print(f"✅ Average inference time per batch: {avg_time:.4f}s | per image: {avg_time / cfg.train.batch_size * 1000:.2f} ms")

if __name__ == "__main__":
    with torch.no_grad():
        main()

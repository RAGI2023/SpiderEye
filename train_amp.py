import os
import time
import yaml
import argparse
from glob import glob
from easydict import EasyDict as edic
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from model.utils.dataset import EquiDataset
from model.E2E import HomoDispNet
from model.utils.tools import *
from model.loss import *


# ------------------- Load Config -------------------
with open('configs/train.yaml') as f:
    g_cfg = edic(yaml.safe_load(f))
    g_cfg.train.lr = float(g_cfg.train.lr)
    g_cfg.train.batch_size = int(g_cfg.train.batch_size)
    g_cfg.train.epochs = int(g_cfg.train.epochs)
    g_cfg.train.num_workers = int(g_cfg.train.num_workers)
    g_cfg.train.weight_decay = float(g_cfg.train.weight_decay)
    g_cfg.model.mean = tuple(map(float, g_cfg.model.mean.split(',')))
    g_cfg.model.std = tuple(map(float, g_cfg.model.std.split(',')))


# ------------------- DDP Setup -------------------
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
        if not os.path.basename(f).startswith("best")
    ]
    ckpts = sorted(ckpts, key=os.path.getmtime)
    return ckpts[-1] if ckpts else None


# ------------------- Main -------------------
def main(args):
    device, local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"✅ Using DDP | world_size={world_size}")

    # 目录
    run_root = os.path.join('runs', g_cfg.experiment.name)
    log_dir = os.path.join(run_root, 'tb')
    ckpt_dir = os.path.join(run_root, 'ckpts')
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Logs -> {log_dir}")

    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    set_seed(g_cfg.data.jitter.random_seed)

    # ---------------- Dataset ----------------
    dataset = EquiDataset(
        folder_path=g_cfg.data.train_dataset,
        fov=g_cfg.data.fov,
        canvas_size=(g_cfg.data.canvas_size[0], g_cfg.data.canvas_size[1]),
        out_w=g_cfg.data.canvas_size[1],
        out_h=g_cfg.data.canvas_size[1],
        jitter_cfg=g_cfg.data.jitter
    )

    sampler = DistributedSampler(dataset, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=g_cfg.train.batch_size,
        sampler=sampler,
        num_workers=g_cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=(g_cfg.train.num_workers > 0),
        drop_last=False
    )

    if rank == 0:
        print(f"Dataset size: {len(dataset)} | Batch size: {g_cfg.train.batch_size}")

    # ---------------- Model ----------------
    net = HomoDispNet(opt=g_cfg.model, device=device)
    # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # 分布式 BN
    net = net.to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    total_params = count_params(net)
    if rank == 0:
        print(f"Model params: {total_params / 1e6:.2f}M")

    optimizer = torch.optim.Adam(net.parameters(), lr=g_cfg.train.lr, weight_decay=g_cfg.train.weight_decay)

    # ----------- AMP 初始化 -----------
    scaler = torch.cuda.amp.GradScaler()

    # ---------------- Resume if needed ----------------
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.continue_train:
        latest_ckpt = get_latest_ckpt(ckpt_dir)
        if latest_ckpt:
            if rank == 0:
                print(f"🔄 Loading latest checkpoint: {latest_ckpt}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            torch.serialization.add_safe_globals([edic])
            ckpt = torch.load(latest_ckpt, map_location=map_location, weights_only=False)
            net.module.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            global_step = ckpt.get('global_step', 0)
            best_loss = ckpt.get('avg_loss', float('inf'))
            if rank == 0:
                print(f"✅ Resumed from epoch {start_epoch}, step {global_step}")

    # ---------------- Train Loop ----------------
    num_epochs = g_cfg.train.epochs
    total_elapsed = 0.0
    log_interval = g_cfg.log.log_interval
    save_interval = g_cfg.log.save_interval
    lambda1 = g_cfg.train.lambda1
    lambda2 = g_cfg.train.lambda2

    l1_charbonnier_loss = L1_Charbonnier_loss(eps=1e-6)

    if rank == 0:
        print("################## Start Training (AMP enabled) #######################")

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        running_loss = 0.0

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=100)

        net.train()

        for i, (imgs, img_original) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            img_original = img_original.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # ---------- Forward + Loss (AMP) ----------
            with torch.cuda.amp.autocast():
                outs = net(imgs)
                loss_l_num = l1_charbonnier_loss(outs, img_original)
                loss_ssim = ssim_loss(outs, img_original, window_size=11, is_train=True)
                loss_gradient = gradient_loss(outs, img_original)
                loss = (1 - lambda1) * loss_l_num + lambda1 * loss_ssim + lambda2 * loss_gradient

            # ---------- Backward (AMP) ----------
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=getattr(g_cfg.train, 'grad_clip', 3))
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            # ---------- TensorBoard ----------
            if rank == 0:
                writer.add_scalar("Loss/L_num", loss_l_num.item(), global_step)
                writer.add_scalar("Loss/SSIM", loss_ssim.item(), global_step)
                writer.add_scalar("Loss/Gradient", loss_gradient.item(), global_step)
                writer.add_scalar("Loss/Total", loss.item(), global_step)

                if global_step % log_interval == 0:
                    vis_front = imgs[0, 0].detach().cpu()
                    vis_right = imgs[0, 1].detach().cpu()
                    vis_back = imgs[0, 2].detach().cpu()
                    vis_left = imgs[0, 3].detach().cpu()
                    vis_out = outs[0].detach().cpu().clamp(0, 1)
                    writer.add_images("Images/Front", vis_front.unsqueeze(0), global_step)
                    writer.add_images("Images/Left", vis_left.unsqueeze(0), global_step)
                    writer.add_images("Images/Back", vis_back.unsqueeze(0), global_step)
                    writer.add_images("Images/Right", vis_right.unsqueeze(0), global_step)
                    writer.add_images("Images/Output", vis_out.unsqueeze(0), global_step)
                    writer.add_images("Images/GroundTruth", img_original[0].detach().cpu().unsqueeze(0), global_step)

                    if net.module.weights is not None:
                        weight_vis = net.module.weights[0]
                        for idx in range(weight_vis.shape[0]):
                            writer.add_image(f"Weights/Direction_{idx}", weight_vis[idx].unsqueeze(0), global_step)

            # ---------- Save ckpt ----------
            if rank == 0 and global_step % save_interval == 0 and global_step > 0:
                ckpt = {
                    'epoch': epoch + 1,
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'avg_loss': loss.item(),
                    'cfg': dict(g_cfg),
                    'global_step': global_step,
                }
                save_ckpt(ckpt, os.path.join(ckpt_dir, f'step_{global_step}.pth'))

            global_step += 1

            if rank == 0:
                pbar.update(1)
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        if rank == 0:
            pbar.close()

        # ---------- 同步 loss ----------
        loss_tensor = torch.tensor(running_loss / len(loader), device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = loss_tensor.item() / world_size

        epoch_time = time.perf_counter() - epoch_start
        total_elapsed += epoch_time

        # ---------- Epoch Summary ----------
        if rank == 0:
            epoch_ips = (len(dataset) / epoch_time)
            writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
            writer.add_scalar('Speed/img_per_sec_epoch', epoch_ips, epoch)

            ckpt = {
                'epoch': epoch + 1,
                'model': net.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'avg_loss': avg_loss,
                'cfg': dict(g_cfg),
                'global_step': global_step,
            }
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_ckpt(ckpt, os.path.join(ckpt_dir, 'best.pth'))

            print(f"✅ Epoch {epoch + 1}/{num_epochs} | AvgLoss: {avg_loss:.4f} | "
                  f"Time: {format_secs(epoch_time)} | Best: {best_loss:.4f}")

    if rank == 0:
        writer.close()
        print(f"Training done. Total time: {format_secs(total_elapsed)}")

    dist.destroy_process_group()


# ------------------- Entry -------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)

import os
import time
import yaml
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


with open('configs/train.yaml') as f:
    g_cfg = edic(yaml.safe_load(f))
    g_cfg.train.lr = float(g_cfg.train.lr)
    g_cfg.train.batch_size = int(g_cfg.train.batch_size)
    g_cfg.train.epochs = int(g_cfg.train.epochs)
    g_cfg.train.num_workers = int(g_cfg.train.num_workers)
    g_cfg.train.weight_decay = float(g_cfg.train.weight_decay)
    g_cfg.model.mean = tuple(map(float, g_cfg.model.mean.split(',')))
    g_cfg.model.std = tuple(map(float, g_cfg.model.std.split(',')))

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank

def main():
    device, local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"✅ Using DDP | world_size={world_size}")

    # 目录
    run_root = os.path.join('runs', g_cfg.experiment.name)
    log_dir  = os.path.join(run_root, 'tb')
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
    net = HomoDispNet(opt=g_cfg.model, device=device).to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    total_params = count_params(net)
    if rank == 0:
        print(f"Model params: {total_params/1e6:.2f}M")

    optimizer = torch.optim.Adam(net.parameters(), lr=g_cfg.train.lr, weight_decay=g_cfg.train.weight_decay)

    num_epochs = g_cfg.train.epochs
    best_loss = float('inf')
    total_elapsed = 0.0
    global_step = 0

    log_interval = g_cfg.log.log_interval
    save_interval = g_cfg.log.save_interval
    
    if rank == 0:
        print("################## Start Training (FP32) #######################")

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        running_loss = 0.0

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        net.train()

        for i, (imgs, img_original) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)  # imgs:(B,6,3,H,W)
            img_original = img_original.to(device, non_blocking=True)  # GT 标签

            # ---------- Forward ----------
            outs = net(imgs)
            # ---------- Loss ----------
            loss = l_num_loss(outs, img_original, num=1)

            # ---------- Backward ----------
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=getattr(g_cfg.train, 'grad_clip', 3))
            optimizer.step()

            running_loss += loss.item()
            global_step += 1

            # ---------- TensorBoard ----------
            if rank == 0:
                
                writer.add_scalar("Loss/Total", loss.item(), global_step)

                # 写可视化图像
                if global_step % log_interval == 0:
                    # 只取 batch 的前 1 张
                    vis_front = imgs[0,0].detach().cpu()
                    vis_right  = imgs[0,1].detach().cpu()
                    vis_back   = imgs[0,2].detach().cpu()
                    vis_left   = imgs[0,3].detach().cpu()
                    vis_out   = outs[0,0].detach().cpu().clamp(0, 1)

                    writer.add_images("Images/Front", vis_front, global_step)
                    writer.add_images("Images/Left", vis_left, global_step)
                    writer.add_images("Images/Back", vis_back, global_step)
                    writer.add_images("Images/Right", vis_right, global_step)
                    writer.add_images("Images/Output", vis_out, global_step)
     

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

        # ---------- Checkpoint ----------
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
            }
            if epoch % save_interval == 0 or (epoch == num_epochs - 1):
                save_ckpt(ckpt, os.path.join(ckpt_dir, f'{epoch}.pth'))
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_ckpt(ckpt, os.path.join(ckpt_dir, 'best.pth'))

            print(f"✅ Epoch {epoch+1}/{num_epochs} | AvgLoss: {avg_loss:.4f} | "
                  f"Time: {format_secs(epoch_time)} | Best: {best_loss:.4f}")

    if rank == 0:
        writer.close()
        print(f"Training done. Total time: {format_secs(total_elapsed)}")
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

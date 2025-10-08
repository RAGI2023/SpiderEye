import yaml
from easydict import EasyDict as edic
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.utils.dataset import ImageFolderDataset
from model.network import SeamNet
from model.utils.tools import *

with open('configs/train.yaml') as f:
    g_cfg = edic(yaml.safe_load(f))
    g_cfg.train.lr = float(g_cfg.train.lr)
    g_cfg.train.batch_size = int(g_cfg.train.batch_size)
    g_cfg.train.epochs = int(g_cfg.train.epochs)
    g_cfg.train.num_workers = int(g_cfg.train.num_workers)



def main():
    print(g_cfg)

    # 目录
    run_root = os.path.join('runs', g_cfg.experiment.name)
    log_dir  = os.path.join(run_root, 'tb')
    ckpt_dir = os.path.join(run_root, 'ckpts')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # 随机种子 & 设备
    set_seed(g_cfg.data.jitter.random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集 & DataLoader
    dataset = ImageFolderDataset(
        folder_path=g_cfg.data.train_dataset,
        fov=g_cfg.data.fov,
        out_w=g_cfg.data.sub_size[0],
        out_h=g_cfg.data.sub_size[1],
        jitter_cfg=g_cfg.data.jitter
    )
    loader = DataLoader(
        dataset,
        batch_size=g_cfg.train.batch_size,
        shuffle=True,
        num_workers=g_cfg.train.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(g_cfg.train.num_workers > 0),
        drop_last=False
    )

    print(f"Dataset size: {len(dataset)} | Batch size: {g_cfg.train.batch_size} | Steps/epoch: {len(loader)}")

    # 模型
    net = SeamNet().to(device)
    total_params = count_params(net)
    print(f"Model params: {total_params/1e6:.2f}M")

    # 优化器 / loss / AMP
    optimizer = torch.optim.Adam(net.parameters(), lr=g_cfg.train.lr)
    criterion = nn.L1Loss()
    use_amp = bool(getattr(g_cfg.train, 'amp', True))
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    scaler = torch.amp.GradScaler(device_type=device_type, enabled=use_amp)


    # 训练循环
    num_epochs = g_cfg.train.epochs
    net.train()

    global_step = 0
    best_loss = float('inf')
    total_elapsed = 0.0  # 累计训练时间(秒)

    print("################## Start Training #######################")

    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        running_loss = 0.0

        pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)

        for i, batch in enumerate(loader):
            batch_start = time.perf_counter()

            imgs = ensure_tensor_imgs(batch).to(device, non_blocking=True)

            # ----- 前向 + 计算损失 -----
            with torch.amp.autocast(device_type=device_type, enabled=use_amp):
                outs = net(imgs[:, 0], imgs[:, 1])
                # 按你的任务需要调整监督信号
                loss = criterion(outs, imgs[:, 5])

            # ----- 反向传播 -----
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if getattr(g_cfg.train, 'grad_clip', None):
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(net.parameters(), g_cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if getattr(g_cfg.train, 'grad_clip', None):
                    nn.utils.clip_grad_norm_(net.parameters(), g_cfg.train.grad_clip)
                optimizer.step()

            # 日志
            running_loss += loss.item()
            writer.add_scalar('Train/Batch_Loss', loss.item(), global_step)
            writer.add_scalar('Train/LR', optimizer.param_groups[0]['lr'], global_step)

            # 速度统计
            batch_time = time.perf_counter() - batch_start
            imgs_this_batch = imgs.size(0)
            ips = imgs_this_batch / batch_time  # images/sec

            # 控制台输出（更详细）
            if (i % max(1, len(loader)//20) == 0) or (i == len(loader)-1):
                msg = (f"[Epoch {epoch+1}/{num_epochs}] "
                       f"[Batch {i+1}/{len(loader)}] "
                       f"Loss: {loss.item():.4f} | "
                       f"BatchTime: {batch_time*1000:.1f} ms | "
                       f"Throughput: {ips:.1f} img/s")
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated() / (1024**3)
                    msg += f" | GPU Mem: {mem:.2f} GB"
                print(msg)

            pbar.set_postfix(loss=f"{loss.item():.4f}", ips=f"{ips:.1f}")
            pbar.update(1)

            global_step += 1

        pbar.close()

        # 每轮统计
        epoch_time = time.perf_counter() - epoch_start
        total_elapsed += epoch_time
        avg_loss = running_loss / len(loader)
        avg_bt = epoch_time / len(loader)
        epoch_ips = (len(dataset) if not getattr(loader, 'sampler', None) else len(loader)*g_cfg.train.batch_size) / epoch_time

        writer.add_scalar('Train/Epoch_Loss', avg_loss, epoch)
        writer.add_scalar('Time/Epoch_seconds', epoch_time, epoch)
        writer.add_scalar('Time/Batch_seconds_avg', avg_bt, epoch)
        writer.add_scalar('Speed/img_per_sec_epoch', epoch_ips, epoch)

        # ETA（按已完成 epoch 的平均时长估算）
        epochs_done = epoch + 1
        mean_epoch_time = total_elapsed / epochs_done
        eta_secs = (num_epochs - epochs_done) * mean_epoch_time

        print(
            f"✅ Epoch {epoch+1}/{num_epochs} "
            f"| AvgLoss: {avg_loss:.4f} "
            f"| EpochTime: {format_secs(epoch_time)} "
            f"| AvgBatchTime: {avg_bt*1000:.1f} ms "
            f"| Throughput: {epoch_ips:.1f} img/s "
            f"| ETA: {format_secs(eta_secs)}"
        )


        # 保存 checkpoint
        ckpt = {
            'epoch': epoch + 1,
            'model': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if scaler.is_enabled() else None,
            'avg_loss': avg_loss,
            'cfg': dict(g_cfg),
        }
        save_ckpt(ckpt, os.path.join(ckpt_dir, 'latest.pth'))
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_ckpt(ckpt, os.path.join(ckpt_dir, 'best.pth'))

        # 明确打印保存信息
        print(f"💾 Saved: latest.pth  | Best: {best_loss:.4f}")

        # 及时flush，保证日志可见
        writer.flush()

    writer.close()
    print(f"Training done. Total time: {format_secs(total_elapsed)}")


if __name__ == '__main__':
    print("Starting training...")
    main()

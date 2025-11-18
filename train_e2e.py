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

from model.utils.EquiDataset import EquiDataset
from model.utils.FishEyeDataset import FishEyeDataset
from model.StitchNet import HomoDispNet
from model.utils.tools import *
from model.loss.common_loss import *
from model.ColorStitchNet import ColorStitchNet

from model.loss.vgg_loss import VGGPerceptualLoss

with open('configs/tuning_fisheye.yaml') as f:
    g_cfg = edic(yaml.safe_load(f))
    check_cfg_keys(g_cfg)

def main(args):
    device, local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print(f"✅ Using DDP | world_size={world_size}")

    # 目录
    run_root = os.path.join('runs', g_cfg.experiment.name)
    log_dir  = os.path.join(run_root, 'tb')
    load_ckpt_dir = os.path.join('runs', g_cfg.experiment.get('load_name', g_cfg.experiment.name), 'ckpts') # priority to load_name
    ckpt_dir = os.path.join('runs', g_cfg.experiment.name, 'ckpts') 
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Logs -> {log_dir}")

    writer = SummaryWriter(log_dir=log_dir) if rank == 0 else None

    set_seed(g_cfg.data.jitter.random_seed)
    torch.backends.cudnn.benchmark = True  # 固定输入尺寸时可加速

    jitter_cfg = {
        "random_seed": g_cfg.data.jitter.random_seed,
        "rotation_jitter": {
            "yaw": g_cfg.data.jitter.yaw,
            "pitch": g_cfg.data.jitter.pitch,
            "roll": g_cfg.data.jitter.roll
        },
        "translate_range": g_cfg.data.jitter.translate,
        "lighting": {
            "brightness": g_cfg.data.jitter.brightness,
            "contrast": g_cfg.data.jitter.contrast,
            "color_jitter": g_cfg.data.jitter.color_jitter
        },
        "k_jitter": g_cfg.data.jitter.k_jitter,
    }

    # ---------------- Dataset ----------------
    dataset_type = g_cfg.data.get('dataset_type', 'panorama')
    if dataset_type == 'fisheye':
        dataset = FishEyeDataset(
            folder_path=g_cfg.data.train_dataset,
            canvas_size=(g_cfg.data.canvas_size[0], g_cfg.data.canvas_size[1]),
            gt_type=g_cfg.data.get('gt_type', 'Samsung')
        )
        eval_dataset = FishEyeDataset(
            g_cfg.data.eval_dataset,
            canvas_size=(g_cfg.data.canvas_size[0], g_cfg.data.canvas_size[1]),
            gt_type=g_cfg.data.get('gt_type', 'Samsung')
        )
    else:
        dataset = EquiDataset(
            folder_path=g_cfg.data.train_dataset,
            fov=g_cfg.data.fov,
            canvas_size=(g_cfg.data.canvas_size[0], g_cfg.data.canvas_size[1]),
            out_w=g_cfg.data.canvas_size[1],
            out_h=g_cfg.data.canvas_size[1],
            jitter_cfg=jitter_cfg,
            k=(0.35, -0.0015, 0.002, -0.002),
        )
        eval_dataset = EquiDataset(
            folder_path=g_cfg.data.eval_dataset,
            fov=g_cfg.data.fov,
            canvas_size=(g_cfg.data.canvas_size[0], g_cfg.data.canvas_size[1]),
            out_w=g_cfg.data.canvas_size[1],
            out_h=g_cfg.data.canvas_size[1],
            jitter_cfg=jitter_cfg,
            k=(00.35, -0.0015, 0.002, -0.002),
        )

    # 训练 & 评估的分布式采样器（评估不 shuffle）
    train_sampler = DistributedSampler(dataset, shuffle=True)
    eval_sampler  = DistributedSampler(eval_dataset, shuffle=False)

    loader = DataLoader(
        dataset,
        batch_size=g_cfg.train.batch_size,
        sampler=train_sampler,
        num_workers=g_cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=(g_cfg.train.num_workers > 0),
        drop_last=False
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=g_cfg.train.batch_size,
        sampler=eval_sampler,
        num_workers=max(1, g_cfg.train.num_workers // 2),
        pin_memory=True,
        persistent_workers=False,
        drop_last=False
    )

    if rank == 0:
        hparams = {
            'lr': g_cfg.train.lr,
            'batch_size': g_cfg.train.batch_size,
            'epochs': g_cfg.train.epochs,
            'weight_decay': g_cfg.train.weight_decay,
            'num_workers': g_cfg.train.num_workers,
            'local_adj_limit': g_cfg.model.local_adj_limit,
            'canvas_size': str(g_cfg.data.canvas_size),
            'lambda1': g_cfg.train.lambda1,
            'lambda2': g_cfg.train.lambda2,
            'lambda3': g_cfg.train.lambda3,
            'lambda4': g_cfg.train.lambda4,
            'l_num': g_cfg.train.l_num,
            'type': g_cfg.model.get('type', 'UNet')
        }
        writer.add_hparams(hparams, {})

    if rank == 0:
        print(f"Dataset size: {len(dataset)} | Batch size: {g_cfg.train.batch_size}")

    # ---------------- Model ----------------
    if g_cfg.model.color_correction:
        net = ColorStitchNet(opt=g_cfg.model, device=device)
    else:
        net = HomoDispNet(opt=g_cfg.model, device=device)

    net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = net.to(device)
    net = nn.parallel.DistributedDataParallel(net, device_ids=[local_rank], output_device=local_rank)

    # loss 模块
    affine_loss_module = FlowIdentityLoss(reduction='mean').to(device)  # 仅保留，若不用可删除
    vgg_loss_module = VGGPerceptualLoss().to(device)
    total_params = count_params(net)

    if rank == 0:
        print(f"Model params: {total_params/1e6:.2f}M")
        print(f'Backbone Type: {g_cfg.model.type}')

    optimizer = torch.optim.Adam(net.parameters(), lr=g_cfg.train.lr, weight_decay=g_cfg.train.weight_decay)

    # ---------------- Resume if needed ----------------
    start_epoch = 0
    global_step = 0
    best_loss = float('inf')

    if args.continue_train:
        latest_ckpt = get_latest_ckpt(load_ckpt_dir)
        if latest_ckpt:
            if rank == 0:
                print(f"🔄 Loading latest checkpoint: {latest_ckpt}")
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            torch.serialization.add_safe_globals([edic])
            ckpt = torch.load(latest_ckpt, map_location=map_location, weights_only=False)
            net.module.load_state_dict(ckpt['model'])
            # optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt.get('epoch', 0)
            last_step = ckpt.get('global_step', 0)
            best_loss = ckpt.get('avg_loss', float('inf'))
            if rank == 0:
                print(f"✅ Reload from {latest_ckpt}")
                print(f"✅ Resumed from epoch {start_epoch}, step {last_step}")

    # ---------------- Train Loop ----------------
    num_epochs = g_cfg.train.epochs
    total_elapsed = 0.0

    log_interval = g_cfg.log.log_interval
    save_interval = g_cfg.log.save_interval
    eval_interval = g_cfg.log.eval_interval

    lambda1 = g_cfg.train.lambda1
    lambda2 = g_cfg.train.lambda2
    lambda3 = g_cfg.train.lambda3
    lambda4 = g_cfg.train.lambda4
    l_num = g_cfg.train.l_num

    if rank == 0:
        print("################## Start Training (FP32) #######################")

    for epoch in range(start_epoch, num_epochs+start_epoch):
        # 分别对 train/eval sampler 设 epoch
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)

        epoch_start = time.perf_counter()
        running_loss = 0.0

        if rank == 0:
            pbar = tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{num_epochs+start_epoch}", ncols=100)

        net.train()

        for i, (imgs, img_original) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            img_original = img_original.to(device, non_blocking=True)

            # ---------- Forward ----------
            outs = net(imgs)

            # ---------- Loss ----------
            # loss_l_num = l1_charbonnier_loss(outs, img_original)
            loss_l_num = l_num_loss(outs, img_original, num=l_num)
            loss_ssim = ssim_loss(outs, img_original, window_size=11, is_train=True)
            loss_gradient = gradient_loss(outs, img_original)
            loss_affine = affine_loss(net.module.theta)  # 按你原始写法保留
            loss_vgg = vgg_loss_module(outs, img_original)
            loss = lambda1 * loss_l_num + lambda2 * loss_ssim + lambda3 * loss_affine + lambda4 * loss_vgg
            
            # ---------- Backward ----------
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=getattr(g_cfg.train, 'grad_clip', 3))
            optimizer.step()

            running_loss += loss.item()

            # ---------- TensorBoard ----------
            if rank == 0:
                # print(f"Step {global_step} | Loss: {loss.item():.4f} | ")
                writer.add_scalar("Loss/L_num", loss_l_num.item(), global_step)
                writer.add_scalar("Loss/SSIM", loss_ssim.item(), global_step)
                writer.add_scalar("Loss/Gradient", loss_gradient.item(), global_step)
                writer.add_scalar("Loss/Affine", loss_affine.item(), global_step)
                writer.add_scalar("Loss/VGG", loss_vgg.item(), global_step)
                writer.add_scalar("Loss/Total", loss.item(), global_step)

                if (global_step % log_interval == 0) and (global_step > 0):
                    vis_front = imgs[0,0].detach().cpu()
                    vis_right = imgs[0,1].detach().cpu()
                    vis_back  = imgs[0,2].detach().cpu()
                    vis_left  = imgs[0,3].detach().cpu()
                    vis_out   = outs[0].detach().cpu().clamp(0, 1)
                    writer.add_images("Images/Front", vis_front.unsqueeze(0), global_step)
                    writer.add_images("Images/Left",  vis_left.unsqueeze(0),  global_step)
                    writer.add_images("Images/Back",  vis_back.unsqueeze(0),  global_step)
                    writer.add_images("Images/Right", vis_right.unsqueeze(0), global_step)
                    writer.add_images("Images/Output", vis_out.unsqueeze(0),  global_step)
                    writer.add_images("Images/GroundTruth", img_original[0].detach().cpu().unsqueeze(0), global_step)

                    # weights
                    if net.module.weights is not None:
                        weight_vis = net.module.weights[0]  # [homography*N, H, W]
                        for idx in range(weight_vis.shape[0]):
                            writer.add_image(f"Weights/Direction_{idx}", weight_vis[idx].unsqueeze(0), global_step)
                        
                    if net.module.record_warped and net.module.warped is not None:
                        for dir_idx, warped_imgs in enumerate(net.module.warped):
                            writer.add_image(f"Warped/Direction_{dir_idx}", warped_imgs, global_step)
                        

            # ---------- Save ckpt per step ----------
            if rank == 0 and (global_step % save_interval == 0) and (global_step > 0):
                ckpt = {
                    'epoch': epoch + 1,
                    'model': net.module.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
                    'avg_loss': loss.item(),
                    'cfg': dict(g_cfg),
                    'global_step': global_step,
                }
                save_ckpt(ckpt, os.path.join(ckpt_dir, f'step_{global_step}.pth'))

            # ---------- Eval（分布式，跳过 step=0）----------
            if (global_step % eval_interval == 0) and (global_step > 0):
                net.eval()
                local_time_sum = 0.0
                local_sample_cnt = 0
                logged_vis = False

                with torch.no_grad():
                    for j, (e_imgs, e_img_original) in enumerate(eval_loader):
                        e_imgs = e_imgs.to(device, non_blocking=True)

                        # torch.cuda.synchronize(device)
                        t0 = time.perf_counter()

                        e_outs = net(e_imgs)

                        # torch.cuda.synchronize(device)
                        t1 = time.perf_counter()

                        local_time_sum += (t1 - t0)
                        local_sample_cnt += e_imgs.size(0)

                        if (rank == 0) and (not logged_vis):
                            vis_out = e_outs[0].detach().cpu().clamp(0, 1)
                            writer.add_images("Eval/Output", vis_out.unsqueeze(0), global_step)
                            writer.add_images("Eval/GroundTruth", e_img_original[0].detach().cpu().unsqueeze(0), global_step)
                            logged_vis = True

                # 汇总各 rank 的时间与样本数
                t_tensor = torch.tensor([local_time_sum], device=device)
                n_tensor = torch.tensor([local_sample_cnt], device=device)
                dist.all_reduce(t_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(n_tensor, op=dist.ReduceOp.SUM)

                total_time = t_tensor.item()
                total_samples = int(n_tensor.item())
                avg_ms_per_image = (total_time / max(total_samples, 1)) * 1000.0

                if rank == 0:
                    writer.add_scalar("Eval/Avg_Infer_ms_per_image", avg_ms_per_image, global_step)
                    print(f"✅ Eval @ step {global_step} | Avg infer time: {avg_ms_per_image:.3f} ms/img "
                          f"(samples={total_samples}, world_size={world_size})")

                net.train()

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

            print(f"✅ Epoch {epoch+1}/{num_epochs} | AvgLoss: {avg_loss:.4f} | "
                  f"Time: {format_secs(epoch_time)} | Best: {best_loss:.4f}")

    if rank == 0:
        writer.close()
        print(f"Training done. Total time: {format_secs(total_elapsed)}")

    dist.destroy_process_group()


# ------------------- Entry -------------------
if __name__ == '__main__':
    args = parse_args()
    main(args)

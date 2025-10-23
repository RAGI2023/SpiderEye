import os
import cv2
import yaml
import torch
import argparse
import torchvision.utils as vutils
from easydict import EasyDict as edic

from model.StitchNet import HomoDispNet
from model.ColorStitchNet import ColorStitchNet
from model.utils.tools import set_seed, count_params

def load_config(cfg_path='configs/eval.yaml'):
    with open(cfg_path) as f:
        g = edic(yaml.safe_load(f))
        # 只保留必要的转型，避免无关键报错
        g.train.batch_size = int(g.train.get('batch_size', 1))
        g.train.num_workers = int(g.train.get('num_workers', 0))
        g.model.mean = tuple(map(float, g.model.get('mean', '0,0,0').split(',')))
        g.model.std  = tuple(map(float, g.model.get('std',  '1,1,1').split(',')))
    return g

def read_img(path):
    """
    读取 -> BGR2RGB -> [0,1] tensor[C,H,W]
    """
    im = cv2.imread(path)
    if im is None:
        raise FileNotFoundError(f"Image not found: {path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = torch.from_numpy(im).float() / 255.0  # [H,W,C]
    im = im.permute(2, 0, 1).contiguous()     # [C,H,W]
    return im

@torch.no_grad()
def infer_one_sample(model, sample_dir, out_dir):
    """
    sample_dir:
      front.jpg, right.jpg, back.jpg, left.jpg （必须）
      gt.jpg （可选）
    保存：
      <basename>_output.png, <basename>_gt.png(如果存在)
    """
    # 读取四个视角；注意顺序需与训练一致：[front, right, back, left]
    front = read_img(os.path.join(sample_dir, 'front.jpg'))
    right = read_img(os.path.join(sample_dir, 'right.jpg'))
    back  = read_img(os.path.join(sample_dir, 'back.jpg'))
    left  = read_img(os.path.join(sample_dir, 'left.jpg'))

    # 组装成 [1,4,3,H,W]
    imgs = torch.stack([left, front, right, back], dim=0)   # [4,3,H,W]
    imgs = imgs.unsqueeze(0).contiguous()                   # [1,4,3,H,W]
    print(f"🔄 inferencing sample: {sample_dir}, input shape: {imgs.shape}")

    imgs = imgs.to(next(model.parameters()).device)

    # 模型前向
    outs, *_ = model(imgs)          # outs: [1,3,H,W] (按你的网络输出)
    out = outs[0].detach().cpu().clamp(0, 1)  # [3,H,W]
    print("✅ inference done. Output shape:", out.shape)

    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(os.path.normpath(sample_dir))
    out_path = os.path.join(out_dir, f"{base}_output.png")
    vutils.save_image(out, out_path)
    print(f"✅ saved: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Minimal eval (no dataloader, no tqdm)")
    ap.add_argument("--cfg", default="configs/eval.yaml", type=str, help="config file")
    ap.add_argument("--ckpt_dir", type=str, default=None, help="override: runs/<load_name>/ckpts")
    ap.add_argument("--ckpt_name", type=str, default="best.pth", help="checkpoint filename")
    ap.add_argument("--samples", nargs="+", required=True, help="one or more sample directories")
    ap.add_argument("--save_dir", type=str, required=True, help="directory to save outputs")
    ap.add_argument("--side", type=int, default=None, help="input side length (default from cfg.data.canvas_size[1])")
    args = ap.parse_args()

    g = load_config(args.cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    set_seed(g.data.jitter.random_seed)

    # 构建模型
    if g.model.color_correction:
        net = ColorStitchNet(opt=g.model, device=device)
    else:
        net = HomoDispNet(opt=g.model, device=device)
    net = net.to(device)
    print(f"Model params: {count_params(net)/1e6:.2f}M")

    # 加载权重
    ckpt_dir = args.ckpt_dir or os.path.join('runs', g.experiment.load_name, 'ckpts')
    ckpt_path = os.path.join(ckpt_dir, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"🔄 Loading checkpoint: {ckpt_path}")

    torch.serialization.add_safe_globals([edic])
    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt['model'])

    # 逐个目录做一次推理并保存结果
    for sample_dir in args.samples:
        infer_one_sample(net, sample_dir, args.save_dir)

if __name__ == "__main__":
    main()

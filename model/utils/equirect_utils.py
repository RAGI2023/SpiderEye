import cv2 as cv
import numpy as np
import math
import os
import random
import torch
import torch.nn.functional as F

# ==========================================================
# 统一的默认扰动配置
# ==========================================================
DEFAULT_JITTER_CONFIG = {
    "random_seed": None,      # 设置为 None 表示每次运行随机；否则固定随机性
    "rotation_jitter": {      # 旋转扰动范围（度）
        "yaw": 5.0,
        "pitch": 3.0,
        "roll": 3.0
    },
    "translate_range": 20.0,  # 平移扰动范围（像素）
    "lighting": {             # 光照扰动参数
        "brightness": 0.2,    # 亮度扰动 [-0.2, 0.2]
        "contrast": 0.2,      # 对比度扰动 [-0.2, 0.2]
        "color_jitter": 0.1   # RGB颜色扰动 [-0.1, 0.1]
    }
}

NO_JITTER_CONFIG = {
    "random_seed": None,
    "rotation_jitter": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "translate_range": 0.0,
    "lighting": {"brightness": 0.0, "contrast": 0.0, "color_jitter": 0.0}
}

# ==========================================================
# 光照扰动模块
# ==========================================================
def apply_lighting_jitter(img, cfg):
    """根据配置对图像进行光照扰动"""
    b_rng = cfg.get("brightness", 0.2)
    c_rng = cfg.get("contrast", 0.2)
    col_rng = cfg.get("color_jitter", 0.1)

    alpha = 1.0 + random.uniform(-c_rng, c_rng)        # 对比度
    beta = 255 * random.uniform(-b_rng, b_rng)         # 亮度
    rgb_gain = np.array([
        1.0 + random.uniform(-col_rng, col_rng),
        1.0 + random.uniform(-col_rng, col_rng),
        1.0 + random.uniform(-col_rng, col_rng)
    ], dtype=np.float32)

    img = img.astype(np.float32) * rgb_gain
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


# ==========================================================
# 主函数：全景投影 + 扰动增强
# ==========================================================
def perspective_projection_diagfov(
    equirect, fov_diag_deg, yaw_deg, pitch_deg, roll_deg,
    out_w=1024, out_h=1024, interpolation=cv.INTER_LINEAR,
    translate=(0.0, 0.0, 0.0),
    jitter_cfg=None
):
    """
    从 equirectangular 全景图中采样，生成给定【对角FOV】和朝向的视图
    带可配置的旋转/平移/光照扰动
    """
    if jitter_cfg is None:
        jitter_cfg = NO_JITTER_CONFIG

    # 设置随机种子（保证可复现）
    seed = jitter_cfg.get("random_seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 提取扰动配置
    rot_jit = jitter_cfg.get("rotation_jitter", {})
    trans_range = jitter_cfg.get("translate_range", 0.0)
    light_cfg = jitter_cfg.get("lighting", {})

    # 旋转扰动
    yaw_deg += random.uniform(-rot_jit.get("yaw", 0.0), rot_jit.get("yaw", 0.0))
    pitch_deg += random.uniform(-rot_jit.get("pitch", 0.0), rot_jit.get("pitch", 0.0))
    roll_deg += random.uniform(-rot_jit.get("roll", 0.0), rot_jit.get("roll", 0.0))

    # 平移扰动
    tx, ty, tz = np.random.uniform(-trans_range, trans_range, 3)
    translate = (tx, ty, tz)

    # 几何计算部分
    H, W = equirect.shape[:2]
    fov_d = math.radians(fov_diag_deg)
    diag = math.sqrt(out_w**2 + out_h**2)

    xs = np.linspace(-out_w/2, out_w/2, out_w, dtype=np.float32)
    ys = np.linspace(-out_h/2, out_h/2, out_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    # 改为鱼眼投影模型：
    r_norm = np.sqrt(xv**2 + yv**2)
    theta = (r_norm / (diag / 2.0)) * (fov_d / 2.0)
    dirs = np.stack([
        np.sin(theta) * (xv / (r_norm + 1e-8)),
        -np.sin(theta) * (yv / (r_norm + 1e-8)),
        np.cos(theta)
    ], axis=-1)

    dirs += np.array(translate, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    yaw, pitch, roll = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    Rx = np.array([[1,0,0],
                   [0,math.cos(pitch),-math.sin(pitch)],
                   [0,math.sin(pitch), math.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[math.cos(yaw),0,math.sin(yaw)],
                [0,1,0],
                [-math.sin(yaw),0,math.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[math.cos(roll),-math.sin(roll),0],
                   [math.sin(roll), math.cos(roll),0],
                   [0,0,1]], dtype=np.float32)
    R = Rz @ Rx @ Ry
    dirs = dirs @ R.T

    X, Y, Z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = np.arctan2(X, Z)
    phi   = np.arcsin(Y)

    map_x = (theta + math.pi) / (2 * math.pi) * W
    map_y = (math.pi / 2 - phi) / math.pi * H
    view = cv.remap(equirect, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation, borderMode=cv.BORDER_WRAP)

    # 光照扰动
    if light_cfg:
        view = apply_lighting_jitter(view, light_cfg)

    return view

import math
import torch
import torch.nn.functional as F
import random
import numpy as np

# ---- 可选：光照扰动函数（PyTorch 实现版） ----
def apply_lighting_jitter_torch(img, light_cfg):
    """
    对输入 torch 图像应用简单光照扰动（亮度/对比度/噪声）。
    img: [B,3,H,W], 0~1 float
    """
    b, c, h, w = img.shape
    out = img.clone()
    if "brightness" in light_cfg:
        br = light_cfg["brightness"]
        delta = (torch.rand(b, 1, 1, 1, device=img.device) * 2 - 1) * br
        out = out + delta
    if "contrast" in light_cfg:
        cr = light_cfg["contrast"]
        factor = 1 + (torch.rand(b, 1, 1, 1, device=img.device) * 2 - 1) * cr
        mean = out.mean(dim=(2,3), keepdim=True)
        out = (out - mean) * factor + mean
    if "noise" in light_cfg:
        nr = light_cfg["noise"]
        noise = torch.randn_like(out) * nr
        out = out + noise
    return out.clamp(0, 1)


# ---- GPU 版本主函数 ----
def perspective_projection_diagfov_gpu(
    equirect, fov_diag_deg, yaw_deg, pitch_deg, roll_deg,
    out_w=1024, out_h=1024, interpolation="bilinear",
    translate=(0.0, 0.0, 0.0),
    jitter_cfg=None,
):
    """
    从 equirectangular 全景图中采样，生成给定【对角FOV】和朝向的视图（GPU 版本）
    equirect: torch.Tensor, shape [B,3,H,W] or [3,H,W], float32, 0~1, CUDA
    返回: torch.Tensor [B,3,out_h,out_w]
    """

    device = equirect.device
    if equirect.dim() == 3:
        equirect = equirect.unsqueeze(0)  # -> [1,3,H,W]
    B, C, H, W = equirect.shape

    if jitter_cfg is None:
        jitter_cfg = {}

    # 随机扰动配置
    seed = jitter_cfg.get("random_seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    rot_jit = jitter_cfg.get("rotation_jitter", {})
    trans_range = jitter_cfg.get("translate_range", 0.0)
    light_cfg = jitter_cfg.get("lighting", {})

    # 旋转扰动
    yaw_deg += random.uniform(-rot_jit.get("yaw", 0.0), rot_jit.get("yaw", 0.0))
    pitch_deg += random.uniform(-rot_jit.get("pitch", 0.0), rot_jit.get("pitch", 0.0))
    roll_deg += random.uniform(-rot_jit.get("roll", 0.0), rot_jit.get("roll", 0.0))

    # 平移扰动
    tx, ty, tz = np.random.uniform(-trans_range, trans_range, 3)
    translate = (tx, ty, tz)

    # === 构建采样射线 ===
    xs = torch.linspace(-out_w/2, out_w/2, out_w, device=device)
    ys = torch.linspace(-out_h/2, out_h/2, out_h, device=device)
    xv, yv = torch.meshgrid(ys, xs, indexing='xy') 
    diag = math.sqrt(out_w**2 + out_h**2)
    fov_d = math.radians(fov_diag_deg)

    r_norm = torch.sqrt(xv**2 + yv**2)
    theta = (r_norm / (diag/2)) * (fov_d/2)

    dirs = torch.stack([
        torch.sin(theta) * (xv / (r_norm + 1e-8)),
        -torch.sin(theta) * (yv / (r_norm + 1e-8)),
        torch.cos(theta)
    ], dim=-1).type(torch.float32)  # [H,W,3]
    dirs = dirs + torch.tensor(translate, device=device, dtype=torch.float32)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # === 构建旋转矩阵 ===
    yaw, pitch, roll = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    dtype = torch.float32

    Rx = torch.tensor([[1,0,0],
                    [0,math.cos(pitch),-math.sin(pitch)],
                    [0,math.sin(pitch), math.cos(pitch)]],
                    device=device, dtype=dtype)

    Ry = torch.tensor([[math.cos(yaw),0,math.sin(yaw)],
                    [0,1,0],
                    [-math.sin(yaw),0,math.cos(yaw)]],
                    device=device, dtype=dtype)

    Rz = torch.tensor([[math.cos(roll),-math.sin(roll),0],
                    [math.sin(roll), math.cos(roll),0],
                    [0,0,1]],
                    device=device, dtype=dtype)
    R = Rz @ Rx @ Ry

    dirs = dirs @ R.T

    # === 转回 equirectangular 坐标 ===
    X, Y, Z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta = torch.atan2(X, Z)
    phi = torch.asin(Y)

    map_x = (theta + math.pi) / (2 * math.pi)
    map_y = (math.pi/2 - phi) / math.pi

    # grid_sample expects [-1,1]
    grid = torch.stack([map_x*2 - 1, map_y*2 - 1], dim=-1)  # [H,W,2]
    grid = grid.unsqueeze(0).repeat(B,1,1,1)

    # === 采样 ===
    mode = "bilinear" if interpolation == "bilinear" else "nearest"
    out = F.grid_sample(equirect, grid, mode=mode, padding_mode='border', align_corners=False)

    # === 光照扰动 ===
    if light_cfg:
        out = apply_lighting_jitter_torch(out, light_cfg)

    return out


# ==========================================================
# 主入口：测试生成六面体视图
# ==========================================================
if __name__ == "__main__":
    img = cv.imread("image.png")
    if img is None:
        raise FileNotFoundError("No Image input")

    views = {
        "front":  (  0,   0, 0),
        "back":   (180,   0, 0),
        "left":   (-90,   0, 0),
        "right":  ( 90,   0, 0),
        "top":    (  0,  90, 0),
        "bottom": (  0, -90, 0),
    }

    # 可修改配置
    cfg = {
        "random_seed": 42,
        "rotation_jitter": {"yaw": 0, "pitch": 0, "roll": 0},
        "translate_range": 0,
        "lighting": {"brightness": 0.3, "contrast": 0.25, "color_jitter": 0.2}
    }

    os.makedirs("runs/diagfov_output", exist_ok=True)

    for name, (yaw, pitch, roll) in views.items():
        out = perspective_projection_diagfov(
            img,
            fov_diag_deg=100,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            out_w=560,
            out_h=560,
            jitter_cfg=cfg
        )
        cv.imwrite(f"runs/diagfov_output/{name}.png", out)
        print(f"saved {name}")

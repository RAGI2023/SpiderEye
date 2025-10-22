import cv2 as cv
import numpy as np
import math
import os
import random

def solve_theta_limit(k, theta_max=math.pi/2):
    """求解使 θ_d = π/2 的最小 θ₀"""
    k1, k2, k3, k4 = k

    def theta_d(theta):
        return theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)

    lo, hi = 0.0, math.pi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if theta_d(mid) < theta_max:
            lo = mid
        else:
            hi = mid
    return lo

# ==========================================================
# 统一的默认扰动配置
# ==========================================================
DEFAULT_JITTER_CONFIG = {
    "random_seed": None,      # 设置为 None 表示每次运行随机；否则固定随机性
    "rotation_jitter": {      # 旋转扰动范围（度）
        "yaw": 3.0,
        "pitch": 3.0,
        "roll": 3.0
    },
    "translate_range": 3.0,  # 平移扰动范围（像素）
    "lighting": {             # 光照扰动参数
        "brightness": 0.2,    # 亮度扰动 [-0.2, 0.2]
        "contrast": 0.2,      # 对比度扰动 [-0.2, 0.2]
        "color_jitter": 0.1,   # RGB颜色扰动 [-0.1, 0.1]
    },
    "k_jitter": [0.01, 0.01, 0.01, 0.01],
}

NO_JITTER_CONFIG = {
    "random_seed": None,
    "rotation_jitter": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "translate_range": 0.0,
    "lighting": {"brightness": 0.0, "contrast": 0.0, "color_jitter": 0.0},
    "k_jitter": [0.0, 0.0, 0.0, 0.0],
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
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta).astype(np.float32)
    return img


def perspective_projection_fisheye(
    equirect,
    fov_diag_deg=180,
    yaw_deg=0,
    pitch_deg=0,
    roll_deg=0,
    out_w=800,
    out_h=800,
    interpolation=cv.INTER_LINEAR,
    translate=(0, 0, 0),
    jitter_cfg=None,
    k=(0.0, 0.0, 0.0, 0.0),   # (k1, k2, k3, k4)
    f=1.0,                     # 等效焦距
):
    """
    从 equirect 全景图生成基于 OpenCV 鱼眼模型的真实鱼眼投影视图
    支持正确的圆形 mask (基于旋转后的入射角 θ)
    """

    # -------------------------------
    # 初始化扰动配置
    # -------------------------------
    if jitter_cfg is None:
        jitter_cfg = {}

    seed = jitter_cfg.get("random_seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rot_jit = jitter_cfg.get("rotation_jitter", {})
    trans_range = jitter_cfg.get("translate_range", 0.0)
    light_cfg = jitter_cfg.get("lighting", {})
    

    # 扰动
    yaw_deg += random.uniform(-rot_jit.get("yaw", 0), rot_jit.get("yaw", 0))
    pitch_deg += random.uniform(-rot_jit.get("pitch", 0), rot_jit.get("pitch", 0))
    roll_deg += random.uniform(-rot_jit.get("roll", 0), rot_jit.get("roll", 0))
    tx, ty, tz = np.random.uniform(-trans_range, trans_range, 3)
    translate = (tx, ty, tz)

    # -------------------------------
    # 网格生成
    # -------------------------------
    H, W = equirect.shape[:2]
    diag = math.sqrt(out_w**2 + out_h**2)
    fov_d = math.radians(fov_diag_deg)

    xs = np.linspace(-out_w / 2, out_w / 2, out_w, dtype=np.float32)
    ys = np.linspace(-out_h / 2, out_h / 2, out_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)
    r = np.sqrt(xv**2 + yv**2)

    # 归一化半径
    r_norm = r / (diag / 2)
    theta = r_norm * (fov_d / 2)  # 入射角 θ

    # -------------------------------
    # 应用 OpenCV 鱼眼模型
    # -------------------------------
    k1, k2, k3, k4 = k
    theta_d = theta * (1 + k1 * theta**2 + k2 * theta**4 + k3 * theta**6 + k4 * theta**8)


    # -------------------------------
    # 构建射线 (相机坐标系下)
    # -------------------------------
    dirs = np.stack([
        np.sin(theta_d) * (xv / (r + 1e-8)),
        -np.sin(theta_d) * (yv / (r + 1e-8)),
        np.cos(theta_d)
    ], axis=-1)

    dirs += np.array(translate, dtype=np.float32)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # -------------------------------
    # 旋转矩阵
    # -------------------------------
    yaw, pitch, roll = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll),  math.cos(roll), 0],
                   [0, 0, 1]], dtype=np.float32)
    R = Rz @ Rx @ Ry
    dirs = dirs @ R.T

    # -------------------------------
    # 球面采样
    # -------------------------------
    X, Y, Z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    theta_equi = np.arctan2(X, Z)
    phi = np.arcsin(Y)

    map_x = (theta_equi + math.pi) / (2 * math.pi) * W
    map_y = (math.pi / 2 - phi) / math.pi * H

    view = cv.remap(equirect, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation, borderMode=cv.BORDER_WRAP)

    view = apply_lighting_jitter(view, light_cfg)

    return view


# ==========================================================
# 主入口：测试生成六面体视图
# ==========================================================
def main_test_views():
    import time
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
        "lighting": {"brightness": 0.3, "contrast": 0.25, "color_jitter": 0.2},
        "k": [0.05, 0.05, 0.05, 0.05],
    }

    os.makedirs("runs/fisheye_realistic", exist_ok=True)
    fisheye_params = (0.08, -0.16, 0.35, -0.26)
    fisheye_params = tuple(
        fisheye_params[i] + random.uniform(-cfg["k"][i], cfg["k"][i])
        for i in range(len(fisheye_params))
    )

    for name, (yaw, pitch, roll) in views.items():
        start_time = time.time()
        out = perspective_projection_fisheye(
            img,
            fov_diag_deg=190,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            out_w=560,
            out_h=560,
            k=fisheye_params,
            jitter_cfg=cfg
        )
        print(f"{name} view done in {time.time() - start_time:.3f} seconds")
        if cv.imwrite(f"runs/fisheye_realistic/{name}.png", out):
            print(f"saved {name}")
        else:
            print(f"failed to save {name}")

def main_test_k():
    img = cv.imread("image.png")
    os.makedirs("runs/fisheye_realistic", exist_ok=True)

    fisheye_params = (0.08, -0.16, 0.35, -0.26)


 
    out = perspective_projection_fisheye(
        img,
        fov_diag_deg=180,
        yaw_deg=0,
        pitch_deg=0,
        roll_deg=0,
        out_w=560,
        out_h=560,
        k=fisheye_params,
    )
    cv.imwrite(f"runs/fisheye_realistic/fisheye_d.png", out)
    print(f"saved fisheye_d.png")

if __name__ == "__main__":
    main_test_views()
    # main_test_k()

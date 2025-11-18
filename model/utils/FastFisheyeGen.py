import cv2 as cv
import numpy as np
import math

# =============================
# Fast Fisheye Projector
# =============================
class FastFisheyeProjector:
    def __init__(self, out_w=560, out_h=560, fov_diag_deg=180, k=(0,0,0,0), lighting_cfg=None):
        self.out_w = out_w
        self.out_h = out_h
        self.fov = math.radians(fov_diag_deg)
        self.k1, self.k2, self.k3, self.k4 = k
        self.lighting = lighting_cfg

        # ---------------------------------
        # 预计算所有静态缓冲：网格 r、theta、方向射线基底
        # ---------------------------------
        self._precompute_mesh()

    def _precompute_mesh(self):
        H, W = self.out_h, self.out_w
        diag = math.sqrt(W*W + H*H)

        xs = np.linspace(-W/2, W/2, W, dtype=np.float32)
        ys = np.linspace(-H/2, H/2, H, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        r = np.sqrt(xv**2 + yv**2)

        # 计算 theta
        r_norm = r / (diag / 2)
        theta = r_norm * (self.fov / 2)

        # OpenCV 鱼眼模式：theta_d
        theta_d = theta * (
            1 + self.k1 * theta**2 + self.k2 * theta**4 +
            self.k3 * theta**6 + self.k4 * theta**8
        )

        # 圆形 mask（fisheye valid region）
        mask = theta <= (self.fov / 2)
        self.mask = mask

        # -----------------------------
        # 基础方向射线 (未旋转)
        # -----------------------------
        eps = 1e-8
        self.dirs_base = np.stack([
            np.sin(theta_d) * (xv / (r+eps)),
            -np.sin(theta_d) * (yv / (r+eps)),
            np.cos(theta_d)
        ], axis=-1).astype(np.float32)

        # meshgrid 复用
        self.xv = xv
        self.yv = yv

    # -------------------------------------------
    # 简单光照扰动
    # -------------------------------------------
    def _apply_lighting(self, img):
        if self.lighting is None:
            return img

        b = self.lighting.get("brightness", 0)
        c = self.lighting.get("contrast", 0)
        color = self.lighting.get("color_jitter", 0)

        alpha = 1 + np.random.uniform(-c, c)
        beta  = 255*np.random.uniform(-b, b)
        gains = np.random.uniform(1-color, 1+color, 3)

        img = img.astype(np.float32) * gains
        return cv.convertScaleAbs(img, alpha=alpha, beta=beta)

    # -------------------------------------------
    # 生成一个固定角度的视图
    # -------------------------------------------
    def render_view(self, equirect, yaw_deg, pitch_deg=0, roll_deg=0):
        H, W = equirect.shape[:2]

        # -------- 旋转矩阵（3×3）--------
        yaw, pitch, roll = map(math.radians, [yaw_deg, pitch_deg, roll_deg])

        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(pitch), -math.sin(pitch)],
            [0, math.sin(pitch),  math.cos(pitch)]
        ], dtype=np.float32)

        Ry = np.array([
            [ math.cos(yaw), 0, math.sin(yaw)],
            [ 0, 1, 0 ],
            [-math.sin(yaw), 0, math.cos(yaw)]
        ], dtype=np.float32)

        Rz = np.array([
            [math.cos(roll), -math.sin(roll), 0],
            [math.sin(roll),  math.cos(roll), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        R = (Rz @ Rx @ Ry).astype(np.float32)

        # --------- 旋转方向射线 ----------
        dirs = self.dirs_base @ R.T

        # 球面坐标
        X, Y, Z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
        theta = np.arctan2(X, Z)
        phi   = np.arcsin(Y)

        # 映射到 equirect
        map_x = ((theta + math.pi) / (2*math.pi)) * W
        map_y = ((math.pi/2 - phi) / math.pi) * H

        view = cv.remap(
            equirect,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            cv.INTER_LINEAR,
            borderMode=cv.BORDER_WRAP
        )

        # 上 mask
        view[~self.mask] = 0

        # 光照扰动
        view = self._apply_lighting(view)

        return view


# =====================================================
# 固定 front / right / left / back 的快速版本
# =====================================================
def generate_4views_fast(equi, k=(0.35, -0.0015, 0.002, -0.002)):
    projector = FastFisheyeProjector(
        out_w=560,
        out_h=560,
        fov_diag_deg=180,
        k=k,
        lighting_cfg=None  # 如果不需要就设 None
    )

    views = {
        "front":  0,
        "right":  90,
        "back":   180,
        "left":  -90
    }

    outputs = {}
    for name, yaw in views.items():
        outputs[name] = projector.render_view(equi, yaw_deg=yaw)

    return outputs

if __name__ == "__main__":
    # 测试代码
    equi_img = cv.imread("image.png")
    if equi_img is None:
        raise FileNotFoundError("No Image input")

    import time
    import os
    t1 = time.perf_counter()
    views = generate_4views_fast(equi_img)
    t2 = time.perf_counter()
    print(f"Rendering took {t2 - t1:.3f} seconds")
    prefix = "runs/fast_fisheye"
    os.makedirs(prefix, exist_ok=True)
    for name, img in views.items():
        cv.imwrite(f"{prefix}/{name}_view.png", img)
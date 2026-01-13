#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Equirectangular panorama -> fisheye views using UCM (Unified Camera Model)

Key points:
- Correct 6 canonical view directions via base_dir (front/back/left/right/top/bottom)
- Optional circular mask to black-out pixels outside the fisheye image circle
  * mask_mode="inscribed" (default): radius = min(W,H)/2
  * mask_mode="diagonal":  radius = sqrt(W^2+H^2)/2
  * mask_mode="none":      no mask
- Keep yaw/pitch/roll + lighting jitter
- Keep deprecated fov_diag_deg fallback ONLY if f_pix not provided (pinhole approximation)

Dependencies:
  pip install numpy opencv-python
"""

import os
import math
import random
import warnings
import numpy as np
import cv2 as cv

# ==========================================================
# Jitter configs
# ==========================================================
DEFAULT_JITTER_CONFIG = {
    "random_seed": None,
    "rotation_jitter": {"yaw": 3.0, "pitch": 3.0, "roll": 3.0},
    "translate_range": 3.0,
    "lighting": {"brightness": 0.2, "contrast": 0.2, "color_jitter": 0.1},
}

NO_JITTER_CONFIG = {
    "random_seed": None,
    "rotation_jitter": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
    "translate_range": 0.0,
    "lighting": {"brightness": 0.0, "contrast": 0.0, "color_jitter": 0.0},
}

# ==========================================================
# Lighting jitter
# ==========================================================
def apply_lighting_jitter(img, cfg):
    b_rng = cfg.get("brightness", 0.2)
    c_rng = cfg.get("contrast", 0.2)
    col_rng = cfg.get("color_jitter", 0.1)

    alpha = 1.0 + random.uniform(-c_rng, c_rng)        # contrast
    beta = 255 * random.uniform(-b_rng, b_rng)         # brightness
    rgb_gain = np.array([
        1.0 + random.uniform(-col_rng, col_rng),
        1.0 + random.uniform(-col_rng, col_rng),
        1.0 + random.uniform(-col_rng, col_rng)
    ], dtype=np.float32)

    img = img.astype(np.float32) * rgb_gain
    img = cv.convertScaleAbs(img, alpha=alpha, beta=beta).astype(np.float32)
    return img

# ==========================================================
# Rotation helpers
# World/camera convention:
#   x=right, y=up, z=forward
# yaw about +y, pitch about +x, roll about +z
# ==========================================================
def R_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg):
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    roll = math.radians(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, math.cos(pitch), -math.sin(pitch)],
                   [0, math.sin(pitch),  math.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[ math.cos(yaw), 0, math.sin(yaw)],
                   [0, 1, 0],
                   [-math.sin(yaw), 0, math.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[math.cos(roll), -math.sin(roll), 0],
                   [math.sin(roll),  math.cos(roll), 0],
                   [0, 0, 1]], dtype=np.float32)

    return (Rz @ Rx @ Ry).astype(np.float32)

def look_at_rotation(dir_vec, up_hint=np.array([0.0, 1.0, 0.0], dtype=np.float32)):
    """
    Return R such that v_world = R @ v_cam and camera forward (0,0,1) maps to dir_vec.
    """
    f = np.array(dir_vec, dtype=np.float32)
    f = f / (np.linalg.norm(f) + 1e-8)

    up = np.array(up_hint, dtype=np.float32)
    if abs(float(np.dot(f, up))) > 0.99:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    r = np.cross(up, f)
    r = r / (np.linalg.norm(r) + 1e-8)
    u = np.cross(f, r)

    # columns are the world basis of cam x,y,z
    return np.stack([r, u, f], axis=1).astype(np.float32)

# ==========================================================
# UCM lifting: pixel -> unit ray in camera frame
# Input y is image-down; convert to camera-up internally.
# ==========================================================
def ucm_pixel_to_ray(x_pix, y_pix, f_pix, xi):
    mx = x_pix / f_pix
    my = y_pix / f_pix

    # image y-down -> camera y-up
    my = -my

    r2 = mx * mx + my * my
    d = np.sqrt(1.0 + (1.0 - xi * xi) * r2)
    omega = (xi + d) / (r2 + 1.0)

    X = omega * mx
    Y = omega * my
    Z = omega - xi

    n = np.sqrt(X * X + Y * Y + Z * Z) + 1e-8
    return X / n, Y / n, Z / n

# ==========================================================
# Build circular mask
# ==========================================================
def build_circular_mask(out_h, out_w, mode="inscribed"):
    """
    mode:
      - "inscribed": radius = min(W,H)/2  (typical fisheye circle, corners black)
      - "diagonal":  radius = sqrt(W^2+H^2)/2 (covers corners too)
      - "none":      no mask (all True)
    """
    mode = str(mode).lower()
    if mode == "none":
        return np.ones((out_h, out_w), dtype=bool)

    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0
    xv, yv = np.meshgrid(np.arange(out_w, dtype=np.float32),
                         np.arange(out_h, dtype=np.float32))
    r_pix = np.sqrt((xv - cx) ** 2 + (yv - cy) ** 2)

    if mode == "diagonal":
        r_max = math.sqrt(out_w ** 2 + out_h ** 2) / 2.0
    else:
        # default inscribed
        r_max = min(out_w, out_h) / 2.0

    return (r_pix <= r_max)

# ==========================================================
# Main conversion: Equirect -> UCM fisheye
# kwargs:
#   xi (float), f_pix (float preferred)
#   fov_diag_deg (deprecated fallback if f_pix None)
#   mask_mode: "inscribed"|"diagonal"|"none"
# ==========================================================
def equirect_to_fisheye_ucm(
    equirect,
    out_w=800,
    out_h=800,
    base_dir=None,
    yaw_deg=0.0,
    pitch_deg=0.0,
    roll_deg=0.0,
    interpolation=cv.INTER_LINEAR,
    jitter_cfg=None,
    **kwargs,
):
    xi = float(kwargs.get("xi", 0.9))
    f_pix = kwargs.get("f_pix", None)
    mask_mode = kwargs.get("mask_mode", "inscribed")

    fov_diag_deg = kwargs.get("fov_diag_deg", None)
    if fov_diag_deg is not None:
        warnings.warn(
            "Argument 'fov_diag_deg' is deprecated for UCM. "
            "Please pass explicit UCM parameters: f_pix and xi. "
            "This fallback uses a pinhole approximation only.",
            DeprecationWarning,
            stacklevel=2,
        )

    if jitter_cfg is None:
        jitter_cfg = {}

    seed = jitter_cfg.get("random_seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    rot_jit = jitter_cfg.get("rotation_jitter", {})
    light_cfg = jitter_cfg.get("lighting", {})

    # rotation jitter
    yaw_deg += random.uniform(-rot_jit.get("yaw", 0), rot_jit.get("yaw", 0))
    pitch_deg += random.uniform(-rot_jit.get("pitch", 0), rot_jit.get("pitch", 0))
    roll_deg += random.uniform(-rot_jit.get("roll", 0), rot_jit.get("roll", 0))

    H, W = equirect.shape[:2]

    # output principal point
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    xv, yv = np.meshgrid(np.arange(out_w, dtype=np.float32),
                         np.arange(out_h, dtype=np.float32))
    x_pix = xv - cx
    y_pix = yv - cy

    # choose f_pix if not provided
    if f_pix is None:
        if fov_diag_deg is None:
            fov_diag_deg = 180.0
            warnings.warn(
                "Neither 'f_pix' nor 'fov_diag_deg' provided. "
                "Fallback to deprecated fov_diag_deg=180. Please pass f_pix for UCM.",
                UserWarning,
                stacklevel=2,
            )
        # Pinhole approx: r = f * tan(theta). Use diagonal half-radius.
        diag = math.sqrt(out_w ** 2 + out_h ** 2)
        r_max = diag / 2.0
        theta_max = math.radians(float(fov_diag_deg)) / 2.0
        f_pix = r_max / max(math.tan(theta_max), 1e-6)

    f_pix = float(f_pix)

    # pixel -> ray (camera)
    Xc, Yc, Zc = ucm_pixel_to_ray(x_pix, y_pix, f_pix, xi)
    dirs_cam = np.stack([Xc, Yc, Zc], axis=-1).astype(np.float32)

    # base direction
    if base_dir is None:
        base_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    R_base = look_at_rotation(np.array(base_dir, dtype=np.float32))
    R_local = R_from_yaw_pitch_roll(yaw_deg, pitch_deg, roll_deg)
    R = (R_base @ R_local).astype(np.float32)

    dirs = dirs_cam @ R.T

    # ray -> equirect
    X, Y, Z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(X, Z)
    lat = np.arcsin(np.clip(Y, -1.0, 1.0))

    map_x = (lon + math.pi) / (2 * math.pi) * W
    map_y = (math.pi / 2 - lat) / math.pi * H

    view = cv.remap(
        equirect,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation,
        borderMode=cv.BORDER_WRAP,
    )

    # lighting jitter
    view = apply_lighting_jitter(view, light_cfg)

    # apply circular mask (the "black outside FOV" behavior)
    mask = build_circular_mask(out_h, out_w, mode=mask_mode)
    view[~mask] = 0

    return np.clip(view, 0, 255).astype(np.uint8)

# ==========================================================
# Demo: generate 6 canonical views (directions correct)
# ==========================================================
def main_test_views():
    import time
    img = cv.imread("image.png")
    if img is None:
        raise FileNotFoundError("No Image input: image.png")

    # Canonical directions (guaranteed correct)
    views = {
        "front":  np.array([ 0.0,  0.0,  1.0], dtype=np.float32),
        "back":   np.array([ 0.0,  0.0, -1.0], dtype=np.float32),
        "left":   np.array([-1.0,  0.0,  0.0], dtype=np.float32),
        "right":  np.array([ 1.0,  0.0,  0.0], dtype=np.float32),
        "top":    np.array([ 0.0,  1.0,  0.0], dtype=np.float32),
        "bottom": np.array([ 0.0, -1.0,  0.0], dtype=np.float32),
    }

    cfg = {
        "random_seed": 42,
        "rotation_jitter": {"yaw": 0, "pitch": 0, "roll": 0},
        "translate_range": 0,
        "lighting": {"brightness": 0.3, "contrast": 0.25, "color_jitter": 0.2},
    }

    os.makedirs("runs/fisheye_realistic", exist_ok=True)

    # UCM params + mask
    ucm_kwargs = dict(
        xi=0.9,
        f_pix=220.0,
        mask_mode="inscribed",   # <<<<<< this is the black-out behavior you asked for
        # mask_mode="diagonal",
        # mask_mode="none",
        # fov_diag_deg=180.0,  # deprecated fallback only
    )

    for name, base_dir in views.items():
        t0 = time.time()
        out = equirect_to_fisheye_ucm(
            img,
            out_w=560,
            out_h=560,
            base_dir=base_dir,
            yaw_deg=0,
            pitch_deg=0,
            roll_deg=0,
            jitter_cfg=cfg,
            **ucm_kwargs,
        )
        print(f"{name} done in {time.time() - t0:.3f}s")
        cv.imwrite(f"runs/fisheye_realistic/{name}.png", out)

if __name__ == "__main__":
    main_test_views()

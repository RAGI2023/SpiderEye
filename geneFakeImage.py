import cv2 as cv
import numpy as np
import math
import os

def perspective_projection_diagfov(equirect, fov_diag_deg, yaw_deg, pitch_deg, roll_deg,
                                   out_w=1024, out_h=1024, interpolation=cv.INTER_LINEAR):
    """
    从 equirectangular 全景图中采样，生成给定【对角FOV】和朝向的视图
    """
    H, W = equirect.shape[:2]
    fov_d = math.radians(fov_diag_deg)

    # 半对角长
    diag = math.sqrt(out_w**2 + out_h**2)
    # 投影面到相机距离
    z = (diag / 2.0) / math.tan(fov_d / 2.0)

    # 像素坐标中心化：[-w/2, w/2], [-h/2, h/2]
    xs = np.linspace(-out_w/2, out_w/2, out_w, dtype=np.float32)
    ys = np.linspace(-out_h/2, out_h/2, out_h, dtype=np.float32)
    xv, yv = np.meshgrid(xs, ys)

    # 构造相机坐标下的射线
    dirs = np.stack([xv, -yv, np.full_like(xv, z)], axis=-1)
    dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True)

    # 旋转矩阵 (yaw, pitch, roll)
    yaw, pitch, roll = map(math.radians, [yaw_deg, pitch_deg, roll_deg])
    Rx = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]],dtype=np.float32)
    Ry = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]],dtype=np.float32)
    Rz = np.array([[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]],dtype=np.float32)
    R = Rz @ Rx @ Ry

    dirs = dirs @ R.T

    X, Y, Z = dirs[...,0], dirs[...,1], dirs[...,2]

    # 球面坐标
    theta = np.arctan2(Z, X)  # [-pi, pi]
    phi   = np.arcsin(Y)      # [-pi/2, pi/2]

    # 转成 equirect 像素坐标
    map_x = (theta + math.pi) / (2*math.pi) * W
    map_y = (math.pi/2 - phi) / math.pi * H

    return cv.remap(equirect, map_x.astype(np.float32), map_y.astype(np.float32),
                    interpolation, borderMode=cv.BORDER_WRAP)

if __name__ == "__main__":
    img = cv.imread("image.jpg")
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

    os.makedirs("diagfov_output", exist_ok=True)

    for name,(yaw,pitch,roll) in views.items():
        out = perspective_projection_diagfov(img, fov_diag_deg=90,
                                             yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll,
                                             out_w=1920, out_h=1080)
        cv.imwrite(f"diagfov_output/{name}.png", out)
        print("saved", name)

from torch.utils.data import Dataset
import os
import torch
import cv2
from model.utils.equirect_utils import perspective_projection_diagfov
import numpy as np
class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None, jitter_cfg=None, **kwargs):
        self.folder_path = folder_path
        self.jitter_cfg = jitter_cfg

        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}.")
        self.transform = transform

        self.fov = kwargs.get("fov", 130)
        self.out_w = kwargs.get("out_w", 1920)
        self.out_h = kwargs.get("out_h", 1080)

        self.VIEWS = {
            "front":  (  0,   0, 0),
            "back":   (180,   0, 0),
            "left":   (-90,   0, 0),
            "right":  ( 90,   0, 0),
            "top":    (  0,  90, 0),
            "bottom": (  0, -90, 0),
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # imgs:(6, H, W, 3), [0,255]
        imgs = [perspective_projection_diagfov(
            img,
            fov_diag_deg=self.fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            out_w=self.out_w,
            out_h=self.out_h,
            jitter_cfg=self.jitter_cfg
        ) for yaw, pitch, roll in self.VIEWS.values()]
        # 转成 tensor
        # imgs:(6, 3, H, W), [0,1]
        imgs = np.stack(imgs, axis=0)
        imgs = torch.from_numpy(imgs).permute(0, 3, 1, 2).float() / 255.0

        if self.transform:
            imgs = self.transform(imgs)
        return imgs, img_path

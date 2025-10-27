import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import re
import random

class FishEyeDataset(Dataset):
    def __init__(self, folder_path, canvas_size=(1920, 960), gt_type='Samsung', max_retry=5):
        super().__init__()
        self.folder_path = folder_path
        self.gt_type = gt_type
        self.canvas_size = canvas_size
        self.max_retry = max_retry

        if canvas_size[0] / 2 != canvas_size[1]:
            raise ValueError("Canvas size must have a 2:1 aspect ratio (width:height).")

        self.dirs = [
            os.path.join(folder_path, d)
            for d in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, d))
        ]
        if len(self.dirs) == 0:
            raise ValueError(f"No subdirectories found in {folder_path}.")

    def __len__(self):
        return len(self.dirs)

    def __getitem__(self, idx):
        """带自动重试机制的 __getitem__"""
        for retry in range(self.max_retry):
            try:
                dir_path = self.dirs[idx]
                image_files = [
                    os.path.join(dir_path, f)
                    for f in os.listdir(dir_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                if len(image_files) < 2:
                    raise ValueError(f"Expected at least 2 images in {dir_path}, found {len(image_files)}")

                image_files.sort()
                img1 = cv2.imread(image_files[0])
                img2 = cv2.imread(image_files[1])
                if img1 is None or img2 is None:
                    raise ValueError(f"Image read failed in {dir_path}")
                # print(f"Loading images from: {image_files[0]}, {image_files[1]}")
                # print(f"img files", image_files)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

                # 分割前后左右视图
                front = img1[:, :img1.shape[1] // 2, :]
                back  = img1[:, img1.shape[1] // 2:, :]
                right = img2[:, :img2.shape[1] // 2, :]
                left  = img2[:, img2.shape[1] // 2:, :] # right half of img2

                view_size = self.canvas_size[1]
                img_front = cv2.resize(front, (view_size, view_size))
                img_back  = cv2.resize(back,  (view_size, view_size))
                img_left  = cv2.resize(left,  (view_size, view_size))
                img_right = cv2.resize(right, (view_size, view_size))

                outs = np.zeros((4, 3, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8)
                view_size_half = self.canvas_size[0] // 2
                view_interval = self.canvas_size[0] // 4

                outs[0, :, :, :view_size_half] = img_front.transpose(2, 0, 1)
                outs[1, :, :, view_interval+1:1+view_interval+view_size_half] = img_right.transpose(2, 0, 1)
                outs[2, :, :, 2*view_interval:2*view_interval+1+view_size_half] = img_back.transpose(2, 0, 1)
                outs[3, :, :, 3*view_interval:] = img_left[:, :self.canvas_size[0]-(3*view_interval), :].transpose(2, 0, 1)
                outs[3, :, :, :view_interval] = img_left[:, -(self.canvas_size[0]-(3*view_interval)):, :].transpose(2, 0, 1)

                imgs = torch.from_numpy(outs).float() / 255.0

                # === 读取 ground truth ===
                subfolders = [
                    os.path.join(dir_path, d)
                    for d in os.listdir(dir_path)
                    if os.path.isdir(os.path.join(dir_path, d))
                ]

                if len(subfolders) != 1:
                    raise ValueError(f"Expected exactly one subfolder for GT in {dir_path}, found {len(subfolders)}")

                gt_dir = subfolders[0]

                # 支持多种后缀
                possible_exts = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
                gt_path = None
                for ext in possible_exts:
                    candidate = os.path.join(gt_dir, f"{self.gt_type}{ext}")
                    if os.path.exists(candidate):
                        gt_path = candidate
                        break

                if gt_path is None:
                    raise FileNotFoundError(f"GT file not found for {self.gt_type} in {gt_dir}")

                gt_img = cv2.imread(gt_path)
                if gt_img is None:
                    raise ValueError(f"Failed to load GT image: {gt_path}")

                gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
                gt_img = cv2.resize(gt_img, (self.canvas_size[0], self.canvas_size[1]))
                gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

                return imgs, gt_tensor

            except Exception as e:
                print(f"⚠️ [Warning] Sample error at idx {idx} ({dir_path}): {e}")
                # 重新随机取一个样本
                idx = random.randint(0, len(self.dirs) - 1)

        # 如果多次重试仍然失败
        raise RuntimeError(f"Failed to load valid sample after {self.max_retry} retries at idx {idx}")


if __name__ == "__main__":
    dataset = FishEyeDataset(folder_path="../dataset_fisheye", canvas_size=(1024, 512))
    print("Dataset size:", len(dataset))

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    write_dir = "runs/test_fisheye_dataset"
    for imgs, gt in loader:
        # print("Imgs shape:", imgs.shape, "GT shape:", gt.shape)
        if write_dir is not None:
            os.makedirs(write_dir, exist_ok=True)
            for i in range(imgs.shape[0]):
                for v in range(imgs.shape[1]):
                    img_np = (imgs[i, v].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(write_dir, f"img_{i}_view{v}.jpg"), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
                gt_np = (gt[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(write_dir, f"gt_{i}.jpg"), cv2.cvtColor(gt_np, cv2.COLOR_RGB2BGR))
        break
    print("Test completed.")

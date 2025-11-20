import os
import cv2
import math
import torch
import numpy as np
from torch.utils.data import Dataset

# ----------------------------------------------------------
# FastFisheyeProjector
# ----------------------------------------------------------
from model.utils.FastFisheyeGen import FastFisheyeProjector


class EquiVideoDataset(Dataset):
    """
    从 mp4 等视频文件中读取 equirect 视频，
    对每个 clip (T 帧) 生成 4 路鱼眼 + 对应等距 GT。

    返回:
        fisheye_clip: [T, 4, 3, Hc, Wc]  (Hc, Wc = canvas_size)
        equi_clip:    [T, 3, Hc, Wc]
    """

    def __init__(
        self,
        video_root,
        clip_len=4,
        stride=1,
        canvas_size=(1920, 960),
        jitter_cfg=None,
        **kwargs,
    ):
        """
        Args:
            video_root : 文件夹，里面是若干 mp4 / avi / mov 视频文件
            clip_len   : 每个样本的帧数 T
            stride     : 在视频帧上滑动的步长
            canvas_size: (W, H) 输出画布大小（和 EquiDataset 对齐）
            jitter_cfg : 同 EquiDataset，可以包含 'k_jitter', 'lighting' 等
            kwargs     : fov, out_w, out_h, k 等
        """
        super().__init__()

        self.video_root = video_root
        self.clip_len = clip_len
        self.stride = stride
        self.canvas_size = canvas_size
        self.jitter_cfg = jitter_cfg

        # ------- fisheye 参数 -------
        self.fov = kwargs.get("fov", 180)        # 对角 FOV
        self.out_w = kwargs.get("out_w", 560)    # 单个鱼眼图宽
        self.out_h = kwargs.get("out_h", 560)    # 单个鱼眼图高
        self.fisheye_params = kwargs.get("k", (0., 0., 0., 0.))

        self.k_jitter = (
            self.jitter_cfg.get("k_jitter", 0.0)
            if self.jitter_cfg is not None else None
        )

        # ------- 快速鱼眼投影引擎 -------
        self.projector = FastFisheyeProjector(
            out_w=self.out_w,
            out_h=self.out_h,
            fov_diag_deg=self.fov,
            k=self.fisheye_params,
            lighting_cfg=self.jitter_cfg.get("lighting", None)
            if self.jitter_cfg else None
        )

        # ------- 四个视角 -------
        self.VIEWS = {
            "front": (0,   0, 0),
            "back":  (180, 0, 0),
            "left":  (-90, 0, 0),
            "right": (90,  0, 0),
        }

        # ----------------------------------------------------------
        # 扫描 mp4 / avi / mov 文件，构建视频列表和 clip 索引
        # ----------------------------------------------------------
        self.video_paths = []
        self.clip_index = []  # list of (video_idx, start_frame)

        exts = (".mp4", ".avi", ".mov", ".mkv")

        for fname in sorted(os.listdir(video_root)):
            if not fname.lower().endswith(exts):
                continue

            vpath = os.path.join(video_root, fname)
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                print(f"[EquiVideoDataset] Warning: cannot open video {vpath}")
                continue

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if frame_count < clip_len:
                continue

            vid_idx = len(self.video_paths)
            self.video_paths.append(vpath)

            # 滑动窗口生成 clip 起点
            for start in range(0, frame_count - clip_len + 1, stride):
                self.clip_index.append((vid_idx, start))

        if len(self.clip_index) == 0:
            raise ValueError(f"No valid clips found under {video_root}")

        # print(
        #     f"[EquiVideoDataset] Found {len(self.video_paths)} videos, "
        #     f"total {len(self.clip_index)} clips, "
        #     f"clip_len={self.clip_len}, stride={self.stride}"
        # )

    def __len__(self):
        return len(self.clip_index)

    # ----------------------------------------------------------
    # 加载一帧：从视频中读取指定 index 的帧
    # ----------------------------------------------------------
    def _load_clip_frames(self, video_path, start, T):
        """
        从 video_path 中读取从 start 开始的连续 T 帧。
        返回列表 [np.ndarray(H,W,3), ...]，RGB 格式。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        # 定位到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(T):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

        cap.release()

        if len(frames) != T:
            # 理论上不该发生，因为我们根据 CAP_PROP_FRAME_COUNT 构建的索引
            raise RuntimeError(
                f"Failed to read {T} frames from {video_path} starting at {start}, "
                f"only got {len(frames)} frames."
            )

        return frames

    # ----------------------------------------------------------
    # 单帧：使用 fast fisheye 生成 4 视角
    # ----------------------------------------------------------
    def _fisheye_4views(self, img):
        # Distortion k 抖动
        if self.k_jitter is not None:
            base_k = np.array(self.fisheye_params)
            strength = np.random.uniform(1 - self.k_jitter, 1 + self.k_jitter)
            k_new = tuple(base_k * strength)
            self.projector.k1, self.projector.k2, self.projector.k3, self.projector.k4 = k_new

        imgs = [
            self.projector.render_view(
                img,
                yaw_deg=yaw, pitch_deg=pitch, roll_deg=roll
            )
            for yaw, pitch, roll in self.VIEWS.values()
        ]
        return imgs

    # ----------------------------------------------------------
    # 把 4 个 fisheye 图像放到大画布上（与 EquiDataset 完全一致）
    # ----------------------------------------------------------
    def _place_on_canvas(self, imgs):
        """
        imgs: list of 4 fisheye views [Hf, Wf, 3]
        返回: np.ndarray [4, 3, Hc, Wc]，即 [view, C, H, W]
        """
        Wc, Hc = self.canvas_size
        view_size = Wc // 2
        view_interval = Wc // 4

        # resize 每个 fisheye 到 view_size×view_size
        imgs = [cv2.resize(im, (view_size, view_size)) for im in imgs]

        outs = np.zeros((len(self.VIEWS), 3, Hc, Wc), dtype=np.uint8)

        # 这里保持逻辑一致：
        outs[0, :, :, :view_size] = imgs[2].transpose(2, 0, 1)  # front
        outs[1, :, :, view_interval + 1: 1 + view_interval + view_size] = \
            imgs[0].transpose(2, 0, 1)  # right
        outs[2, :, :, 2 * view_interval: 2 * view_interval + 1 + view_size] = \
            imgs[3].transpose(2, 0, 1)  # back

        # left 需要 wrap 一下
        outs[3, :, :, 3 * view_interval:] = \
            imgs[1][:, :Wc - (3 * view_interval), :].transpose(2, 0, 1)
        outs[3, :, :, :view_interval] = \
            imgs[1][:, -(Wc - (3 * view_interval)):, :].transpose(2, 0, 1)

        return outs

    # ----------------------------------------------------------
    # __getitem__：返回一个 clip
    # ----------------------------------------------------------
    def __getitem__(self, idx):
        video_idx, start_frame = self.clip_index[idx]
        vpath = self.video_paths[video_idx]

        # 1) 读取一个 clip 的 T 帧
        frames = self._load_clip_frames(vpath, start_frame, self.clip_len)

        fisheye_clips = []
        equi_clips = []

        Wc, Hc = self.canvas_size

        for img in frames:
            # ---- 鱼眼 4 视角 ----
            four_views = self._fisheye_4views(img)
            canvas = self._place_on_canvas(four_views)  # [4,3,Hc,Wc]
            fisheye_clips.append(torch.from_numpy(canvas).float() / 255.0)

            # ---- 等距 GT resize ----
            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3,H,W]
            equi_resized = torch.nn.functional.interpolate(
                img_t.unsqueeze(0),
                size=(Hc, Wc),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)
            equi_clips.append(equi_resized)

        fisheye_clips = torch.stack(fisheye_clips, dim=0)  # [T,4,3,Hc,Wc]
        equi_clips = torch.stack(equi_clips, dim=0)        # [T,3,Hc,Wc]

        return fisheye_clips, equi_clips


# ----------------------------------------------------------
# 简单自测
# ----------------------------------------------------------
if __name__ == "__main__":
    dataset = EquiVideoDataset(
        video_root=".",  
        clip_len=4,
        stride=1,
        canvas_size=(1024, 512),
        jitter_cfg={
            "k_jitter": 0.1,
            "lighting": {
                "brightness": 0.0,
                "contrast": 0.0,
                "color_jitter": 0.1  # 如果你的 FastFisheyeProjector 用得到
            }
        },
        out_w=560,
        out_h=560,
        k=(0.30, -0.0015, 0.002, -0.002),
        fov=180,
    )

    from torch.utils.data import DataLoader
    import os
    prefix = 'runs/equi_video_dataset_test'
    os.makedirs(prefix, exist_ok=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

    for i, (fisheye_clip, equi_clip) in enumerate(loader):
        print("Clip", i)
        print("fisheye_clip:", fisheye_clip.shape)  # [B, T, 4, 3, Hc, Wc]
        for view in range(4):
            cv2.imwrite(
                f"{prefix}/fisheye_view{view}_clip{i}.png",
                (fisheye_clip[0, 0, view].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
        print("equi_clip:   ", equi_clip.shape)     # [B, T, 3, Hc, Wc]
        cv2.imwrite(
            f"{prefix}/equi_clip{i}.png",
            (equi_clip[0, 0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
        break

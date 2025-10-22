from torch.utils.data import Dataset
import os
import torch
import cv2
try:
    from equirect_utils import perspective_projection_fisheye, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG
except ImportError:
    from model.utils.equirect_utils import perspective_projection_fisheye, DEFAULT_JITTER_CONFIG, NO_JITTER_CONFIG
import numpy as np

class EquiDataset(Dataset):
    def __init__(self, folder_path, canvas_size=(1920, 960), jitter_cfg=None, **kwargs):
        self.folder_path = folder_path
        self.jitter_cfg = jitter_cfg
        self.canvas_size = canvas_size

        self.image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in {folder_path}.")

        self.fov = kwargs.get("fov", 130)
        self.out_w = kwargs.get("out_w", 1920)
        self.out_h = kwargs.get("out_h", 1080)

        self.fisheye_params = kwargs.get("k", (0., 0., 0., 0.))
        self.k_jitter = self.jitter_cfg.get("k_jitter", [0.0, 0.0, 0.0, 0.0]) if self.jitter_cfg else None


        self.VIEWS = {
            "front":  (  0,   0, 0),
            "back":   (180,   0, 0),
            "left":   (-90,   0, 0),
            "right":  ( 90,   0, 0),
            # "top":    (  0,  90, 0),
            # "bottom": (  0, -90, 0),
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        """Get a batch of images for training.
        Returns:
            imgs: Tensor, [4, 3, H, W], [0,1] front, right, back, left
            originally [3, H, W], [0,255]
        """
        img_path = self.image_files[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.k_jitter is not None:
            self.fisheye_params = tuple(
                self.fisheye_params[i] + np.random.uniform(-self.k_jitter[i], self.k_jitter[i])
                for i in range(len(self.fisheye_params))
            )
        # print('Using fisheye params:', self.fisheye_params)
        # imgs:(4, H, W, 3), [0,255]
        imgs = [perspective_projection_fisheye(
            img,
            fov_diag_deg=self.fov,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
            out_w=self.out_w,
            out_h=self.out_h,
            jitter_cfg=self.jitter_cfg,
            k=self.fisheye_params,

        ) for yaw, pitch, roll in self.VIEWS.values()]
        
        # each fisheye img size
        view_size = self.canvas_size[0] // 2
        view_interval = self.canvas_size[0] // 4
        imgs = [cv2.resize(im, (view_size, view_size)) for im in imgs]
        # put into canvas
        outs = np.zeros((len(self.VIEWS), 3, self.canvas_size[1], self.canvas_size[0]), dtype=np.uint8) # [4, 3, H, W]
        outs[0, :, :, :view_size] = imgs[2].transpose(2, 0, 1)  # front
        outs[1, :, :, view_interval+1:1+view_interval+view_size] = imgs[0].transpose(2, 0, 1)  # right
        outs[2, :, :, 2*view_interval:2*view_interval+1+view_size] = imgs[3].transpose(2, 0, 1)  # back
        # special handling for left because of oversize
        outs[3, :, :, 3*view_interval:] = imgs[1][:, :self.canvas_size[0]- (3*view_interval), :].transpose(2, 0, 1)
        outs[3, :, :, :view_interval] = imgs[1][:, -(self.canvas_size[0]- (3*view_interval)):, :].transpose(2, 0, 1)


        # 转成 tensor
        # imgs:(4, 3, H, W), [0,1]
        imgs = torch.from_numpy(outs).float() / 255.0

        #resize img
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W]
        # 2) interpolate 需要 4D，所以加 batch 维；size 接受 (H, W)
        Hc, Wc = self.canvas_size[1], self.canvas_size[0]  # 注意翻转
        img_original = torch.nn.functional.interpolate(
            img_t.unsqueeze(0), size=(Hc, Wc), mode='bilinear', align_corners=False
        ).squeeze(0).to(imgs.device)  # [3, Hc, Wc]
        return imgs, img_original 

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    K = (0.01, -0.1, 0.1, -0.0)
    # K = (0.0, 0.0, 0.0, 0.0)
    dataset = EquiDataset(folder_path="../360SP-data/panoramas", 
                        fov=254, canvas_size=(1024, 512), out_w=512, out_h=512, k=K, jitter_cfg=DEFAULT_JITTER_CONFIG)
    print('Dataset length:', len(dataset))
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    write_img = True

    for i, (imgs, img_original) in enumerate(loader):
        print('Batch', i, 'images shape:', imgs.shape)
        if write_img:
            dirname = 'runs/test_output/'
            os.makedirs(dirname, exist_ok=True)
            imgs = imgs.numpy()
            for b in range(imgs.shape[0]):
                for v, view in enumerate(dataset.VIEWS.keys()):
                    cv2.imwrite(
                        f'{dirname}test_{view}_{i}_{b}.png',
                        cv2.cvtColor((imgs[b, v] * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                    )
            cv2.imwrite(
                f'{dirname}test_original_{i}.png',
                cv2.cvtColor((img_original[0].numpy() * 255).astype(np.uint8).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            )
        break  # 只测试一批

    print('Data loading test completed.')
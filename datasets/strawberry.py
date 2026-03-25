import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .build import DATASETS


@DATASETS.register_module()
class StrawberryDataset(Dataset):
    def __init__(self, config):
        super().__init__()

        # ===== 从config读取 =====
        self.root = config.DATA_PATH
        self.split = config.subset

        self.num_points_partial = config.N_POINTS
        self.num_points_gt = config.GT_POINTS

        # ===== 加载数据路径 =====
        self.files = sorted(glob.glob(
            os.path.join(self.root, self.split, '*.npy')
        ))

        print(f"[{self.split}] dataset loaded: {len(self.files)} samples")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        data = np.load(path, allow_pickle=True).item()

        partial = data["partial"]
        gt = data["gt"]

        # ===== 采样（保证点数一致）=====
        partial = self.random_sample(partial, self.num_points_partial)
        gt = self.random_sample(gt, self.num_points_gt)

        partial = torch.from_numpy(partial).float()
        gt = torch.from_numpy(gt).float()
        taxonomy_id = "strawberry"
        model_id = idx

        return taxonomy_id, model_id, (partial, gt)

    def random_sample(self, pc, n):
        N = pc.shape[0]

        if N == n:
            return pc
        elif N > n:
            idx = np.random.choice(N, n, replace=False)
            return pc[idx]
        else:
            idx = np.random.choice(N, n - N, replace=True)
            return np.concatenate([pc, pc[idx]], axis=0)
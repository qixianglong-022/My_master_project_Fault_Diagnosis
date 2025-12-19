import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from config import Ch4Config


class Ch4DualStreamDataset(Dataset):
    def __init__(self, config: Ch4Config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.cls_map = {'HH': 0, 'RU': 1, 'RM': 2, 'SW': 3, 'VU': 4, 'BR': 5, 'KA': 6, 'FB': 7}

        # 容器
        self.micro, self.macro, self.acoustic = [], [], []
        self.speed, self.labels, self.loads = [], [], []

        # 1. 加载数据
        self._load_data()

        # 2. 预处理流水线
        # Step A: Log 变换 (压缩动态范围)
        self.micro = np.log1p(self.micro)
        self.macro = np.log1p(self.macro)

        # Step B: 归一化 (防泄露核心)
        self.scaler_path = os.path.join(config.CHECKPOINT_DIR, "scaler_ch4_dual.pkl")
        self._handle_normalization()

    def _load_data(self):
        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('_dual.npy')]

        for f in files:
            # Parse: HH_2_1_dual.npy
            parts = f.replace('_dual.npy', '').split('_')
            domain, lid, sid = parts[0], int(parts[1]), parts[2]

            # Map IDs to Physical Values
            phys_load = {0: 0, 2: 200, 4: 400}.get(lid, 0)
            spd_code = self.config.SPEED_ID_MAP.get(sid, 'unknown')

            if (phys_load in target_loads) and (spd_code in target_speeds):
                d = np.load(os.path.join(self.config.DATA_DIR, f), allow_pickle=True).item()
                if len(d['micro']) == 0: continue

                self.micro.append(d['micro'])  # [N, 512]
                self.macro.append(d['panorama'])  # [N, 512]
                self.acoustic.append(d['acoustic'])  # [N, 26]
                self.speed.append(d['speed'])  # [N] (Hz)

                N = len(d['micro'])
                self.labels.append(np.full(N, self.cls_map.get(domain, 0)))
                self.loads.append(np.full(N, d['load']))

        if len(self.micro) > 0:
            self.micro = np.concatenate(self.micro).astype(np.float32)
            self.macro = np.concatenate(self.macro).astype(np.float32)
            self.acoustic = np.concatenate(self.acoustic).astype(np.float32)
            self.speed = np.concatenate(self.speed).astype(np.float32)
            self.labels = np.concatenate(self.labels).astype(np.int64)
            self.loads = np.concatenate(self.loads).astype(np.float32)
        else:
            raise ValueError(f"No data loaded for mode={self.mode}. Check Config path or filtering logic.")

    def _handle_normalization(self):
        if self.mode == 'train':
            # 仅在训练集计算统计量
            scaler = {
                'micro_mean': np.mean(self.micro, axis=0),
                'micro_std': np.std(self.micro, axis=0) + 1e-6,
                'macro_mean': np.mean(self.macro, axis=0),
                'macro_std': np.std(self.macro, axis=0) + 1e-6,
                'ac_mean': np.mean(self.acoustic, axis=0),
                'ac_std': np.std(self.acoustic, axis=0) + 1e-6
            }
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"[Scaler] Fit & Saved to {self.scaler_path}")
        else:
            # 测试集必须加载
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"[Error] Scaler not found at {self.scaler_path}. Please run TRAIN mode first!")
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # 应用 Z-Score
        self.micro = (self.micro - scaler['micro_mean']) / scaler['micro_std']
        self.macro = (self.macro - scaler['macro_mean']) / scaler['macro_std']
        self.acoustic = (self.acoustic - scaler['ac_mean']) / scaler['ac_std']

    def __getitem__(self, idx):
        # 1. 频谱特征 [512, 1] - 增加通道维
        mic = torch.from_numpy(self.micro[idx]).unsqueeze(-1)
        mac = torch.from_numpy(self.macro[idx]).unsqueeze(-1)

        # 2. 声纹特征 [26]
        ac = torch.from_numpy(self.acoustic[idx])

        # 3. 物理转速 (Hz) - 保持 Scalar，供 PGFA 使用
        spd = torch.tensor([self.speed[idx]], dtype=torch.float32)

        # 4. 负载标签 (Regression Target) - 归一化到 0-1
        # 假设最大负载 400kg
        ld = torch.tensor([self.loads[idx] / 400.0], dtype=torch.float32)

        # 5. 分类标签
        lb = torch.tensor(self.labels[idx], dtype=torch.long)

        return mic, mac, ac, spd, lb, ld

    def __len__(self):
        return len(self.labels)
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

        self.micro, self.macro, self.acoustic, self.speed, self.labels, self.loads = [], [], [], [], [], []

        # 1. 加载数据到内存
        self._load_data()

        # 2. 归一化逻辑 (防泄露核心)
        self.scaler_path = os.path.join(config.CHECKPOINT_DIR, "scaler_ch4.pkl")
        self._apply_normalization()

    def _load_data(self):
        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('_dual.npy')]

        for f in files:
            parts = f.replace('_dual.npy', '').split('_')
            domain, lid, sid = parts[0], int(parts[1]), parts[2]

            phys_load = {0: 0, 2: 200, 4: 400}.get(lid, 0)
            spd_code = self.config.SPEED_ID_MAP.get(sid)

            if (phys_load in target_loads) and (spd_code in target_speeds):
                d = np.load(os.path.join(self.config.DATA_DIR, f), allow_pickle=True).item()
                if len(d['micro']) == 0: continue

                self.micro.append(d['micro'])  # [N, 512]
                self.macro.append(d['panorama'])  # [N, 512]
                self.acoustic.append(d['acoustic'])  # [N, 26]
                self.speed.append(d['speed'])  # [N]

                N = len(d['micro'])
                self.labels.append(np.full(N, self.cls_map.get(domain, 0)))
                self.loads.append(np.full(N, d['load']))

        # Concat
        if self.micro:
            self.micro = np.concatenate(self.micro)
            self.macro = np.concatenate(self.macro)
            self.acoustic = np.concatenate(self.acoustic)
            self.speed = np.concatenate(self.speed)
            self.labels = np.concatenate(self.labels)
            self.loads = np.concatenate(self.loads)

            # Log Transform first (for spectra)
            self.micro = np.log1p(self.micro)
            self.macro = np.log1p(self.macro)
        else:
            print(f"[Warn] No data for {self.mode}")

    def _apply_normalization(self):
        if len(self.micro) == 0: return

        if self.mode == 'train':
            # 计算统计量
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
            print(f"[Scaler] Fitting done. Saved to {self.scaler_path}")
        else:
            # 加载统计量
            if os.path.exists(self.scaler_path):
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
            else:
                print("[Warn] Scaler not found! Using raw data (Leads to poor performance).")
                scaler = None

        # 应用变换 (Z-Score)
        if scaler:
            self.micro = (self.micro - scaler['micro_mean']) / scaler['micro_std']
            self.macro = (self.macro - scaler['macro_mean']) / scaler['macro_std']
            self.acoustic = (self.acoustic - scaler['ac_mean']) / scaler['ac_std']

    def __getitem__(self, idx):
        # 转换为 Tensor
        mic = torch.FloatTensor(self.micro[idx]).unsqueeze(-1)  # [512, 1]
        mac = torch.FloatTensor(self.macro[idx]).unsqueeze(-1)
        ac = torch.FloatTensor(self.acoustic[idx])
        spd = torch.FloatTensor([self.speed[idx]])  # Scalar [1]
        lb = torch.LongTensor([self.labels[idx]]).squeeze()

        # Load 归一化: 0~400 -> 0~1
        ld = torch.FloatTensor([self.loads[idx] / 400.0])

        return mic, mac, ac, spd, lb, ld

    def __len__(self):
        return len(self.labels)
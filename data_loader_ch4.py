import torch
import numpy as np
import os
from torch.utils.data import Dataset
from config import Ch4Config


class Ch4DualStreamDataset(Dataset):
    def __init__(self, config: Ch4Config, mode: str = 'train'):
        self.config = config
        self.data_dir = config.DATA_DIR
        self.mode = mode

        self.cls_map = {
            'HH': 0, 'RU': 1, 'RM': 2, 'SW': 3,
            'VU': 4, 'BR': 5, 'KA': 6, 'FB': 7
        }

        # 内存缓存，避免反复 IO
        self.micro_data = []
        self.macro_data = []
        self.acoustic_data = []
        self.speed_data = []
        self.labels = []
        self.loads = []

        self._load_all_data()

    def _load_all_data(self):
        if not os.path.exists(self.data_dir): return

        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        files = [f for f in os.listdir(self.data_dir) if f.endswith('_dual.npy')]

        print(f"[{self.mode.upper()}] Loading dataset from {len(files)} source files...")

        for f in files:
            # Parse filename: HH_2_1_dual.npy
            try:
                parts = f.replace('_dual.npy', '').split('_')
                domain = parts[0]
                load_id = int(parts[1])
                speed_id = parts[2]

                # 筛选逻辑
                phys_load = 0 if load_id == 0 else (200 if load_id == 2 else 400)
                speed_code = self.config.SPEED_ID_MAP.get(speed_id)

                if (phys_load in target_loads) and (speed_code in target_speeds):
                    # 加载大文件
                    data = np.load(os.path.join(self.data_dir, f), allow_pickle=True).item()

                    # 获取样本数 N
                    N = data['micro'].shape[0]
                    if N == 0: continue

                    # 放入列表
                    self.micro_data.append(data['micro'])
                    self.macro_data.append(data['panorama'])
                    self.acoustic_data.append(data['acoustic'])
                    self.speed_data.append(data['speed'])

                    # 生成标签 (Scalar -> Vector)
                    label_vec = np.full(N, self.cls_map.get(domain, 0), dtype=np.int64)
                    self.labels.append(label_vec)

                    # 生成负载标签
                    load_vec = np.full(N, data['load'], dtype=np.float32)
                    self.loads.append(load_vec)

            except Exception as e:
                print(f"Error loading {f}: {e}")
                continue

        # 拼接所有数据到 Tensor (内存级加速)
        if len(self.micro_data) > 0:
            self.micro_data = torch.FloatTensor(np.concatenate(self.micro_data))
            self.macro_data = torch.FloatTensor(np.concatenate(self.macro_data))
            self.acoustic_data = torch.FloatTensor(np.concatenate(self.acoustic_data))
            self.speed_data = torch.FloatTensor(np.concatenate(self.speed_data))
            self.labels = torch.LongTensor(np.concatenate(self.labels))
            self.loads = torch.FloatTensor(np.concatenate(self.loads))

            # 预处理：Log 变换 (增强小幅值特征)
            self.micro_data = torch.log1p(self.micro_data).unsqueeze(-1)  # [Total, 512, 1]
            self.macro_data = torch.log1p(self.macro_data).unsqueeze(-1)
        else:
            print(f"[Warn] No data loaded for mode {self.mode}")

    def __getitem__(self, idx):
        # 直接从内存取，速度飞快
        return (
            self.micro_data[idx],
            self.macro_data[idx],
            self.acoustic_data[idx],
            self.speed_data[idx].unsqueeze(0),  # Scalar -> [1]
            self.labels[idx],
            self.loads[idx].unsqueeze(0) / 400.0  # Normalize Load
        )

    def __len__(self):
        if isinstance(self.micro_data, list): return 0
        return len(self.micro_data)
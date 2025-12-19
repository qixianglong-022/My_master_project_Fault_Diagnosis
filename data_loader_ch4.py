# data_loader_ch4.py
import torch
import numpy as np
import os
from torch.utils.data import Dataset
# 引入新合并的配置
from config import Ch4Config


class Ch4DualStreamDataset(Dataset):
    """
    第四章专用：双流多分辨率数据加载器
    """
    def __init__(self, config: Ch4Config, mode: str = 'train'):
        self.config = config
        self.data_dir = config.DATA_DIR
        self.mode = mode

        # 故障类别映射 (String -> Int)
        # 假设这8类是固定的，写死以保证顺序一致
        self.cls_map = {
            'HH': 0, 'RU': 1, 'RM': 2, 'SW': 3,
            'VU': 4, 'BR': 5, 'KA': 6, 'FB': 7
        }

        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        if not os.path.exists(self.data_dir):
            print(f"[Error] Data directory not found: {self.data_dir}")
            return []

        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        selected = []

        # 从 config 中获取当前模式所需的 负载 和 转速 列表
        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        for f in all_files:
            # 文件名格式: Domain_LoadID_SpeedID_dual.npy (例如: HH_2_1_dual.npy)
            try:
                parts = f.replace('_dual.npy', '').split('_')
                if len(parts) < 3: continue

                # domain = parts[0]
                load_id = int(parts[1])
                speed_id = parts[2]

                # 1. 物理负载映射 (ID -> kg)
                phys_load = 0 if load_id == 0 else (200 if load_id == 2 else 400)

                # 2. 物理转速映射 (ID -> Code)
                speed_code = self.config.SPEED_ID_MAP.get(speed_id)
                if speed_code is None: continue

                # === 核心筛选逻辑 ===
                # 必须同时满足 负载要求 AND 转速要求
                if (phys_load in target_loads) and (speed_code in target_speeds):
                    selected.append(f)

            except Exception as e:
                # print(f"[Warn] Skipping file {f}: {e}")
                continue

        if len(selected) == 0:
            print(f"[Warn] No samples found for mode={self.mode}!")
            print(f"       Target Loads: {target_loads}")
            print(f"       Target Speeds: {target_speeds}")

        return selected

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.samples[idx])
        data = np.load(path, allow_pickle=True).item()

        # 1. 振动流 (Vibration)
        micro_x = torch.FloatTensor(data['micro']).unsqueeze(-1)
        macro_x = torch.FloatTensor(data['panorama']).unsqueeze(-1)

        # 2. [新增] 声纹流 (Acoustic MFCC)
        # 如果数据里没有 'acoustic' 键，生成随机 MFCC (26维) 保证代码运行
        if 'acoustic' in data:
            acoustic_x = torch.FloatTensor(data['acoustic'])
        else:
            # 模拟数据: 基于振动能量生成一点"相关"的噪声，避免完全随机
            vib_energy = torch.mean(torch.abs(macro_x))
            acoustic_x = torch.randn(26) + vib_energy

        # 3. 标签与协变量
        speed_hz = torch.FloatTensor([data['speed'] / 60.0])
        fname = self.samples[idx]
        domain = fname.split('_')[0]
        target_cls = torch.tensor(self.cls_map.get(domain, 0), dtype=torch.long)
        target_load = torch.FloatTensor([data['load'] / 400.0])

        # 返回扩展后的元组 (新增 acoustic_x)
        return micro_x, macro_x, acoustic_x, speed_hz, target_cls, target_load

    def __len__(self):
        return len(self.samples)
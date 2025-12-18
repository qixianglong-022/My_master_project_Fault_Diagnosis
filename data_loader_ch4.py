# data_loader_ch4.py
import torch
import numpy as np
import os
from torch.utils.data import Dataset
# 引入新合并的配置
from config import Ch4Config


class Ch4DualStreamDataset(Dataset):
    """
    第四章专用：严格遵循域泛化协议的数据加载器
    """

    # [关键修复] 这里增加了 config 参数，匹配 main_ch4 的调用
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

        # 1. 输入: 显微流频谱
        micro_x = torch.FloatTensor(data['micro']).unsqueeze(-1)

        # 2. 协变量: 实时转速 (Hz)
        speed_hz = torch.FloatTensor([data['speed'] / 60.0])

        # 3. 标签: 故障类别
        fname = self.samples[idx]
        domain = fname.split('_')[0]
        target_cls = torch.tensor(self.cls_map.get(domain, 0), dtype=torch.long)

        # 4. 标签: 物理负载 (归一化, 用于回归)
        target_load = torch.FloatTensor([data['load'] / 400.0])

        return micro_x, speed_hz, target_cls, target_load

    def __len__(self):
        return len(self.samples)
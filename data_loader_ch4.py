import torch
import numpy as np
import os
from torch.utils.data import Dataset
from config import Config


class Ch4DualStreamDataset(Dataset):
    """
    第四章专用：双流频域数据加载器
    自动关联显微流(Micro)与全景流(Panorama)
    """

    def __init__(self, mode='train'):
        self.data_dir = os.path.join(Config.PROJECT_ROOT, "processed_data_ch4_dual_stream")
        self.mode = mode
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        all_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        # 自动化域泛化逻辑：根据负载划分训练/测试集
        selected = []
        for f in all_files:
            # 文件名格式: Domain_LoadID_SpeedID_dual.npy
            load_id = int(f.split('_')[1])
            phys_load = 0 if load_id == 0 else (200 if load_id == 2 else 400)

            if self.mode == 'train' and phys_load in Config.CH4_TRAIN_LOADS:
                selected.append(f)
            elif self.mode == 'test' and phys_load in Config.CH4_TEST_LOADS:
                selected.append(f)
        return selected

    def __getitem__(self, idx):
        data = np.load(os.path.join(self.data_dir, self.samples[idx]), allow_pickle=True).item()

        # micro_x: [Freq_Dim, 1], panorama_x: [Freq_Dim, 1]
        micro_x = torch.FloatTensor(data['micro']).unsqueeze(-1)
        # 归一化负载标签 (用于虚拟传感器回归)
        target_load = torch.FloatTensor([data['load'] / 400.0])
        # 故障类别标签 (0-7)
        domain_str = self.samples[idx].split('_')[0]
        domain_list = list(Config.DATA_DOMAINS.keys())
        target_cls = torch.tensor(domain_list.index(domain_str))
        # 实时转速转为转频 (Hz)
        speed_hz = torch.FloatTensor([data['speed'] / 60.0])

        return micro_x, speed_hz, target_cls, target_load

    def __len__(self):
        return len(self.samples)
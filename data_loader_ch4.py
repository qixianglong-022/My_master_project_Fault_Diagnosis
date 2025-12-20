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

        # 数据容器
        self.micro, self.macro, self.acoustic, self.current_spec = [], [], [], []
        self.speed_hz, self.current_rms = [], []  # 物理协变量
        self.labels = []

        # 辅助：保留物理标签用于评估分析 (例如区分 0kg 和 400kg 精度)
        self.phys_load_labels = []

        # 1. 加载数据
        self._load_data()

        # 2. 预处理流水线
        # Log 变换 (压缩频谱动态范围)
        self.micro = np.log1p(self.micro)
        self.macro = np.log1p(self.macro)
        # 电流频谱也建议做 Log 变换，因为它也符合长尾分布
        self.current_spec = np.log1p(self.current_spec)

        # 3. 归一化 (核心：Soft Sensing 逻辑)
        self.scaler_path = os.path.join(config.CHECKPOINT_DIR, "scaler_ch4_soft.pkl")
        self._handle_normalization()

    def _load_data(self):
        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('_dual.npy')]

        for f in files:
            # Parse: HH_2_1_dual.npy -> Domain, LoadID, SpeedID
            parts = f.replace('_dual.npy', '').split('_')
            domain, lid, sid = parts[0], int(parts[1]), parts[2]

            # 映射物理值用于筛选
            phys_load_kg = {0: 0, 2: 200, 4: 400}.get(lid, 0)
            spd_code = self.config.SPEED_ID_MAP.get(sid, 'unknown')

            if (phys_load_kg in target_loads) and (spd_code in target_speeds):
                d = np.load(os.path.join(self.config.DATA_DIR, f), allow_pickle=True).item()
                if len(d['micro']) == 0: continue

                self.micro.append(d['micro'])  # [N, 512]
                self.macro.append(d['macro'])  # [N, 512]
                self.acoustic.append(d['acoustic'])  # [N, 15]
                self.current_spec.append(d['current'])  # [N, 128] 频谱

                # 物理引导变量
                self.speed_hz.append(d['speed'])  # [N] 转速频率
                self.current_rms.append(d['load_rms'])  # [N] 电流有效值 (Soft Load Proxy)

                N = len(d['micro'])
                label_str = d.get('label_domain', parts[0])
                self.labels.append(np.full(N, self.cls_map.get(label_str, 0)))
                self.phys_load_labels.append(np.full(N, phys_load_kg))  # 仅用于分析，不进模型

        if len(self.micro) > 0:
            self.micro = np.concatenate(self.micro).astype(np.float32)
            self.macro = np.concatenate(self.macro).astype(np.float32)
            self.acoustic = np.concatenate(self.acoustic).astype(np.float32)
            self.current_spec = np.concatenate(self.current_spec).astype(np.float32)
            self.speed_hz = np.concatenate(self.speed_hz).astype(np.float32)
            self.current_rms = np.concatenate(self.current_rms).astype(np.float32)
            self.labels = np.concatenate(self.labels).astype(np.int64)
            self.phys_load_labels = np.concatenate(self.phys_load_labels).astype(np.int64)
        else:
            raise ValueError(f"No data loaded for mode={self.mode}.")

    def _handle_normalization(self):
        if self.mode == 'train':
            scaler = {
                'micro_mean': np.mean(self.micro, axis=0),
                'micro_std': np.std(self.micro, axis=0) + 1e-6,
                'macro_mean': np.mean(self.macro, axis=0),
                'macro_std': np.std(self.macro, axis=0) + 1e-6,
                'ac_mean': np.mean(self.acoustic, axis=0),
                'ac_std': np.std(self.acoustic, axis=0) + 1e-6,
                'curr_spec_mean': np.mean(self.current_spec, axis=0),
                'curr_spec_std': np.std(self.current_spec, axis=0) + 1e-6,

                # [核心修改] 统计电流RMS的最大值，用于 Soft Sensing 归一化
                # 使用 Max 归一化保留 "0 RMS = 0 Energy" 的物理意义
                'curr_rms_max': np.max(self.current_rms) + 1e-6
            }
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"[Scaler] Soft Sensing params fit & saved to {self.scaler_path}")
        else:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"[Error] Scaler missing. Run TRAIN first.")
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)

        # 应用 Z-Score 标准化 (特征)
        self.micro = (self.micro - scaler['micro_mean']) / scaler['micro_std']
        self.macro = (self.macro - scaler['macro_mean']) / scaler['macro_std']
        self.acoustic = (self.acoustic - scaler['ac_mean']) / scaler['ac_std']
        self.current_spec = (self.current_spec - scaler['curr_spec_mean']) / scaler['curr_spec_std']

        # 应用 MinMax 归一化 (物理引导变量)
        # 将 Current RMS 映射到 [0, 1] 区间，作为 Load Proxy
        self.current_rms = self.current_rms / scaler['curr_rms_max']
        # 截断保护 (防止测试集电流超过训练集最大值太多)
        self.current_rms = np.clip(self.current_rms, 0.0, 1.2)

    def __getitem__(self, idx):
        # 1. 特征流
        mic = torch.from_numpy(self.micro[idx]).unsqueeze(-1)  # [512, 1]
        mac = torch.from_numpy(self.macro[idx]).unsqueeze(-1)  # [512, 1]
        ac = torch.from_numpy(self.acoustic[idx])  # [15]
        cur_spec = torch.from_numpy(self.current_spec[idx])  # [128]

        # 2. 物理引导变量 (Covariates)
        # Speed: 保持 Hz 数值，供 PGFA 计算频率掩码
        spd = torch.tensor([self.speed_hz[idx]], dtype=torch.float32)

        # Load Proxy: 归一化后的 Current RMS，供 Trend 分支感知能量水平
        ld_proxy = torch.tensor([self.current_rms[idx]], dtype=torch.float32)

        # 3. 标签
        # lb: 故障分类标签
        # phys_load: 真实物理负载 (仅用于评估时拆分 0kg/400kg 指标)
        lb = torch.tensor(self.labels[idx], dtype=torch.long)
        phys_load = torch.tensor(self.phys_load_labels[idx], dtype=torch.float32)

        # 返回 7 个元素，保持与 main loop 兼容
        # 注意：这里把 phys_load 藏在原来的 ld 位置返回吗？
        # 不，训练器里 forward 需要的是 input tensor。
        # 所以我们返回 ld_proxy 给模型，phys_load 仅作为 metadata 没法直接通过这就返回
        # 我们可以复用 ld_proxy 位置返回给模型，模型 forward 接受它。
        # 至于评估时的物理负载拆分，由于我们已经归一化了，可以反推：
        #   Real_RMS = ld_proxy * scaler_max
        #   然后根据 RMS 大小判断是轻载还是重载。

        return mic, mac, ac, cur_spec, spd, ld_proxy, lb

    def __len__(self):
        return len(self.labels)
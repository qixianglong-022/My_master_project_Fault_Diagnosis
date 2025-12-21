import torch
import numpy as np
import os
import pickle
from torch.utils.data import Dataset
from config import Ch4Config


class Ch4DualStreamDataset(Dataset):
    """
    [Chapter 4 Dataset]
    特性：双流频谱(Micro/Macro) + 物理引导(Current/Speed)
    修正：移除查表校准，使用 Current RMS 相对值作为 Soft Load Proxy
    """

    def __init__(self, config: Ch4Config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.cls_map = {'HH': 0, 'RU': 1, 'RM': 2, 'SW': 3, 'VU': 4, 'BR': 5, 'KA': 6, 'FB': 7}

        # 数据容器
        self.micro, self.macro, self.acoustic, self.current_spec = [], [], [], []
        self.speed_hz, self.current_rms = [], []
        self.labels = []
        self.phys_load_labels = []  # 仅用于评估阶段拆分指标 (0kg vs 400kg)

        # 1. 加载数据
        self._load_data()

        # 2. 对数变换 (压缩长尾分布)
        self.micro = np.log1p(self.micro)
        self.macro = np.log1p(self.macro)
        self.current_spec = np.log1p(self.current_spec)

        # 3. 物理归一化 (核心修改点)
        self.scaler_path = os.path.join(config.CHECKPOINT_DIR, "scaler_ch4_soft.pkl")
        self._handle_normalization()

    def _load_data(self):
        target_loads = self.config.TRAIN_LOADS if self.mode == 'train' else self.config.TEST_LOADS
        target_speeds = self.config.TRAIN_SPEEDS if self.mode == 'train' else self.config.TEST_SPEEDS

        # 扫描目录
        if not os.path.exists(self.config.DATA_DIR):
            raise FileNotFoundError(f"Data dir not found: {self.config.DATA_DIR}")

        files = [f for f in os.listdir(self.config.DATA_DIR) if f.endswith('_dual.npy')]

        for f in files:
            # 解析文件名: HH_2_1_dual.npy
            try:
                parts = f.replace('_dual.npy', '').split('_')
                domain, lid, sid = parts[0], int(parts[1]), parts[2]
            except:
                continue

            # 简单的物理映射用于筛选文件 (非校准用途)
            phys_load_kg = {0: 0, 2: 200, 4: 400}.get(lid, -1)
            spd_code = self.config.SPEED_ID_MAP.get(sid, 'unknown')

            if (phys_load_kg in target_loads) and (spd_code in target_speeds):
                d = np.load(os.path.join(self.config.DATA_DIR, f), allow_pickle=True).item()
                if len(d['micro']) == 0: continue

                self.micro.append(d['micro'])  # [N, 512]
                self.macro.append(d['macro'])  # [N, 512]
                self.acoustic.append(d['acoustic'])  # [N, 15]
                self.current_spec.append(d['current'])  # [N, 128]

                self.speed_hz.append(d['speed'])  # [N]
                self.current_rms.append(d['load_rms'])  # [N] 原始电流RMS

                N = len(d['micro'])
                self.labels.append(np.full(N, self.cls_map.get(domain, 0)))
                self.phys_load_labels.append(np.full(N, phys_load_kg))

        if len(self.micro) > 0:
            # 转换为 Numpy 数组
            self.micro = np.concatenate(self.micro).astype(np.float32)
            self.macro = np.concatenate(self.macro).astype(np.float32)
            self.acoustic = np.concatenate(self.acoustic).astype(np.float32)
            self.current_spec = np.concatenate(self.current_spec).astype(np.float32)
            self.speed_hz = np.concatenate(self.speed_hz).astype(np.float32)
            self.current_rms = np.concatenate(self.current_rms).astype(np.float32)
            self.labels = np.concatenate(self.labels).astype(np.int64)
            self.phys_load_labels = np.concatenate(self.phys_load_labels).astype(np.int64)
        else:
            print(f"[Warn] No data found for {self.mode}. Check DATA_DIR.")

    def _handle_normalization(self):
        """
        [Anti-Calibration Logic]
        不使用查表，而是统计 Training Set (200kg) 的最大电流值作为基准。
        """
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

                # [关键] 记录训练集(200kg)下的最大电流RMS
                'curr_rms_ref': np.max(self.current_rms) + 1e-6
            }
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"[Scaler] Fit on Source Domain. Max Current Ref: {scaler['curr_rms_ref']:.4f}")
        else:
            if not os.path.exists(self.scaler_path):
                # 如果没有 Scaler，为了不报错，临时算一个 (仅供调试)
                print("[Warn] Scaler missing! Calculating temporary stats.")
                scaler = {'micro_mean': 0, 'micro_std': 1, 'macro_mean': 0, 'macro_std': 1,
                          'ac_mean': 0, 'ac_std': 1, 'curr_spec_mean': 0, 'curr_spec_std': 1,
                          'curr_rms_ref': np.max(self.current_rms)}
            else:
                with open(self.scaler_path, 'rb') as f:
                    scaler = pickle.load(f)

        # 1. 特征标准化 (Z-Score)
        self.micro = (self.micro - scaler['micro_mean']) / scaler['micro_std']
        self.macro = (self.macro - scaler['macro_mean']) / scaler['macro_std']
        self.acoustic = (self.acoustic - scaler['ac_mean']) / scaler['ac_std']
        self.current_spec = (self.current_spec - scaler['curr_spec_mean']) / scaler['curr_spec_std']

        # 2. 物理量归一化 (Soft Sensing)
        # 将电流映射为相对负载率: Load_Proxy = Current / Current_Ref
        # 200kg -> ~1.0
        # 0kg   -> ~0.4
        # 400kg -> ~1.8 (允许超过1.0，体现重载特性)
        self.current_rms = self.current_rms / scaler['curr_rms_ref']

        # [关键] 放宽 Clip 范围，允许 Extrapolation (外推)
        # 之前 clip(1.2) 太保守了，400kg 可能会被截断，导致模型无法区分 300kg 和 400kg
        # 放宽到 3.0 足够覆盖电机过载情况
        self.current_rms = np.clip(self.current_rms, 0.0, 3.0)

    def __getitem__(self, idx):
        # 特征流
        mic = torch.from_numpy(self.micro[idx]).unsqueeze(-1)  # [512, 1]
        mac = torch.from_numpy(self.macro[idx]).unsqueeze(-1)  # [512, 1]
        ac = torch.from_numpy(self.acoustic[idx])  # [15]
        cur_spec = torch.from_numpy(self.current_spec[idx])  # [128]

        # 物理引导变量
        spd = torch.tensor([self.speed_hz[idx]], dtype=torch.float32)
        ld_proxy = torch.tensor([self.current_rms[idx]], dtype=torch.float32)

        # 标签
        lb = torch.tensor(self.labels[idx], dtype=torch.long)

        # [新增] 真实的物理负载标签 (仅用于评估分桶)
        phys_load = torch.tensor(self.phys_load_labels[idx], dtype=torch.float32)

        # 返回 8 个元素 (末尾追加 phys_load)
        return mic, mac, ac, cur_spec, spd, ld_proxy, lb, phys_load

    def __len__(self):
        return len(self.labels)
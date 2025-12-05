# data_loader.py
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from config import Config


class MotorDataset(Dataset):
    def __init__(self, flag='train', atoms=None):
        """
        :param flag: 'train' / 'val' / 'test'
        :param atoms: list of (Load, Speed). 如果为 None，则根据 flag 自动从 Config 读取。
        """
        self.window_size = Config.WINDOW_SIZE

        # 1. 确定要加载的原子列表
        if atoms is not None:
            target_atoms = atoms
        else:
            # 如果没传，根据 flag 决定默认行为
            if flag == 'train' or flag == 'val':
                target_atoms = Config.TRAIN_ATOMS
            else:
                target_atoms = Config.TEST_ATOMS  # 默认测试集

        self.target_atoms = target_atoms
        self.flag = flag

        # 2. 加载数据
        # 训练集只加载 HH，测试集需要同时支持 HH 和 FB 的逻辑
        # 这里为了通用，我们默认加载 HH。如果是测试 FB，需要在外部通过专门的类或方法区分。
        # 简化方案：增加一个 condition 参数，默认为 'HH'
        self.features, self.speeds = self._load_atoms(condition='HH')

        # 3. 标准化 (Z-Score)
        self._apply_scaler()

        # 4. 内部划分 (Train/Val Split)
        self._split_data()

    def _load_atoms(self, condition):
        """加载指定工况列表的所有 .npy 文件"""
        data_list = []
        speed_list = []

        for load, speed in self.target_atoms:
            # 文件名格式: HH_2_1.npy
            fname_x = f"{condition}_{load}_{speed}.npy"
            fname_s = f"{condition}_{load}_{speed}_S.npy"

            path_x = os.path.join(Config.ATOMIC_DATA_DIR, fname_x)
            path_s = os.path.join(Config.ATOMIC_DATA_DIR, fname_s)

            if os.path.exists(path_x):
                x = np.load(path_x)  # [N, 21]
                s = np.load(path_s)  # [N, 1]
                data_list.append(x)
                speed_list.append(s)
            else:
                # 容错：如果找不到，静默跳过或打印警告
                # print(f"[Warn] Missing atom: {fname_x}")
                pass

        if not data_list:
            return np.empty((0, Config.ENC_IN)), np.empty((0, 1))

        return np.concatenate(data_list, axis=0), np.concatenate(speed_list, axis=0)

    def set_condition(self, condition):
        """【关键】用于测试时切换 HH/FB"""
        # 重新加载数据
        self.features, self.speeds = self._load_atoms(condition)
        # 重新标准化
        self._apply_scaler()
        # 重新计算长度
        self._split_data()  # Reset to full length

    def _apply_scaler(self):
        # 统一使用 Source Domain (Load 2) 的参数
        # 假设训练脚本会生成这个文件
        scaler_path = os.path.join(Config.OUTPUT_DIR, "source_scaler.pkl")

        if self.flag == 'train' and not os.path.exists(scaler_path):
            # 如果是第一次训练，计算并保存
            self.mean = np.mean(self.features, axis=0)
            self.std = np.std(self.features, axis=0) + 1e-6
            # 确保目录存在
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump({'mean': self.mean, 'std': self.std}, f)
        else:
            # 其他情况（Val, Test），加载参数
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    params = pickle.load(f)
                self.mean = params['mean']
                self.std = params['std']
            else:
                # 兜底
                self.mean = 0
                self.std = 1

        self.features = (self.features - self.mean) / self.std

    def _split_data(self):
        total = len(self.features)
        if total == 0:
            self.n_samples = 0
            return

        if self.flag == 'train':
            # 80% 用于训练
            split = int(total * 0.8)
            self.features = self.features[:split]
            self.speeds = self.speeds[:split]
            self.stride = 1
        elif self.flag == 'val':
            # 20% 用于验证
            split = int(total * 0.8)
            self.features = self.features[split:]
            self.speeds = self.speeds[split:]
            self.stride = self.window_size  # 不重叠
        else:
            # Test 全量
            self.stride = self.window_size

        if len(self.features) > self.window_size:
            self.n_samples = (len(self.features) - self.window_size) // self.stride + 1
        else:
            self.n_samples = 0

    def __getitem__(self, index):
        start_idx = index * self.stride
        end_idx = start_idx + self.window_size

        # 预测偏移 (Next-Step Prediction)
        PREDICT_STEP = 3

        # 边界保护
        if end_idx + PREDICT_STEP > len(self.features):
            start_idx = len(self.features) - self.window_size - PREDICT_STEP
            end_idx = start_idx + self.window_size

        x_win = self.features[start_idx:end_idx]
        y_win = self.features[start_idx + PREDICT_STEP: end_idx + PREDICT_STEP]
        s_win = self.speeds[start_idx:end_idx]

        v_bar = np.mean(s_win)
        v2_bar = np.mean(s_win ** 2)
        cov = np.array([v_bar / 3000.0, v2_bar / (3000.0 ** 2)], dtype=np.float32)

        return torch.from_numpy(x_win).float(), torch.from_numpy(y_win).float(), torch.from_numpy(cov).float()

    def __len__(self):
        return self.n_samples
import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from config import Config


class MotorDataset(Dataset):
    def __init__(self, atoms_list, mode='train', fault_types=None, noise_snr=None):
        """
        :param atoms_list: list of (Load, Speed)
        :param mode: 'train' (HH only), 'test' (HH + Faults)
        :param noise_snr: float (dB) or None. 仅在测试模式生效。
        """
        self.window_size = Config.WINDOW_SIZE
        self.predict_step = Config.PREDICT_STEP
        self.mode = mode
        self.atoms_list = atoms_list
        self.noise_snr = noise_snr  # <--- 接收接口参数

        # 1. 确定 Domain
        if mode == 'train':
            self.domains = ['HH']
        else:
            target_faults = fault_types if fault_types else Config.TEST_FAULT_TYPES
            self.domains = ['HH'] + target_faults

        # 2. 加载数据
        self.x, self.s, self.y, self.labels = self._load_data()

        # 3. 归一化
        self._apply_scaler()

        # 4. 计算长度
        self.stride = Config.STRIDE // Config.HOP_LENGTH if mode == 'train' else self.window_size
        self.n_samples = (len(self.x) - self.window_size - self.predict_step) // self.stride + 1
        if self.n_samples < 0: self.n_samples = 0

    def _load_data(self):
        xs, ss, ys, ls = [], [], [], []

        for domain in self.domains:
            # 简单标签: HH=0, 其他=1
            label = 0 if domain == 'HH' else 1

            for (load, speed) in self.atoms_list:
                fn_x = f"{domain}_{load}_{speed}.npy"
                fn_s = f"{domain}_{load}_{speed}_S.npy"
                path_x = os.path.join(Config.ATOMIC_DATA_DIR, fn_x)
                path_s = os.path.join(Config.ATOMIC_DATA_DIR, fn_s)

                if os.path.exists(path_x):
                    x_data = np.load(path_x)
                    s_data = np.load(path_s)

                    # === 噪声注入接口 ===
                    # 仅在测试模式且 noise_snr 不为 None 时触发
                    if self.mode == 'test' and self.noise_snr is not None:
                        x_data = self._add_noise(x_data, self.noise_snr)

                    xs.append(x_data)
                    ss.append(s_data)
                    ys.append(x_data)  # 自监督: Target = Input
                    ls.append(np.ones(len(x_data)) * label)

        if not xs: return np.empty((0, Config.ENC_IN)), np.empty((0, 1)), [], []
        return np.concatenate(xs), np.concatenate(ss), np.concatenate(ys), np.concatenate(ls)

    def _add_noise(self, data, snr_db):
        """
        向声纹通道注入高斯白噪声
        """
        # 假设声纹在最后几列 (根据 Config 逻辑)
        # 这里为了简单，对所有通道注入，或者你可以只对 Audio 注入
        # 既然是环境噪声，通常振动和声纹都会受影响，但声纹更明显。
        # 这里的实现是对全通道加噪：
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        return data + noise

    def _apply_scaler(self):
        if self.mode == 'train':
            self.mean = np.mean(self.x, axis=0)
            self.std = np.std(self.x, axis=0) + 1e-6
            os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)
            with open(Config.SCALER_PATH, 'wb') as f:
                pickle.dump({'mean': self.mean, 'std': self.std}, f)
        else:
            if os.path.exists(Config.SCALER_PATH):
                with open(Config.SCALER_PATH, 'rb') as f:
                    p = pickle.load(f)
                    self.mean, self.std = p['mean'], p['std']
            else:
                self.mean, self.std = 0, 1

        self.x = (self.x - self.mean) / self.std
        self.y = (self.y - self.mean) / self.std

    def __getitem__(self, idx):
        i = idx * self.stride
        x_win = self.x[i: i + self.window_size]
        s_win = self.s[i: i + self.window_size]
        y_win = self.y[i + self.predict_step: i + self.window_size + self.predict_step]

        v = np.mean(s_win)
        cov = np.array([v, v ** 2], dtype=np.float32)

        return torch.FloatTensor(x_win), torch.FloatTensor(y_win), torch.FloatTensor(cov), self.labels[i]

    def __len__(self):
        return self.n_samples
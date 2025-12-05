import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from config import Config


class MotorDataset(Dataset):
    def __init__(self, atoms_list, mode='train', fault_types=None, noise_snr=None):
        """
        :param atoms_list: list of (Load, Speed) 这里的 Load/Speed 是物理值 (如 200, '15')
        :param mode: 'train' (HH only), 'test' (HH + Faults)
        :param noise_snr: float (dB) or None. 仅在测试模式生效。
        """
        self.window_size = Config.WINDOW_SIZE
        self.predict_step = Config.PREDICT_STEP
        self.mode = mode
        self.atoms_list = atoms_list
        self.noise_snr = noise_snr

        # 1. 确定 Domain
        if mode == 'train':
            self.domains = ['HH']
        else:
            target_faults = fault_types if fault_types else Config.TEST_FAULT_TYPES
            self.domains = ['HH'] + target_faults

        # 2. 加载数据
        self.x, self.s, self.y, self.labels = self._load_data()

        # 3. (已移除全局归一化，保持原始物理意义交给 RevIN)
        pass

        # 4. 计算长度
        self.stride = Config.STRIDE // Config.HOP_LENGTH if mode == 'train' else self.window_size
        self.n_samples = (len(self.x) - self.window_size - self.predict_step) // self.stride + 1
        if self.n_samples < 0: self.n_samples = 0

    def _load_data(self):
        xs, ss, ys, ls = [], [], [], []

        # [新增] 用于 Debug 的计数器
        missing_files_count = 0
        total_files_expected = 0

        for domain in self.domains:
            # 简单标签: HH=0, 其他=1
            label = 0 if domain == 'HH' else 1

            for (phys_load, phys_speed) in self.atoms_list:
                total_files_expected += 1

                # ================= 核心修正 =================
                # 使用 Config 中的映射表，将物理值转回文件 ID
                # 物理值: 200 -> ID: '2'
                # 物理值: '15' -> ID: '1'
                try:
                    load_id = Config.LOAD_MAP[phys_load]
                    speed_id = Config.SPEED_MAP[phys_speed]
                except KeyError as e:
                    print(f"[Config Error] 无法在 LOAD_MAP/SPEED_MAP 中找到键值: {e}")
                    print(f"    当前请求工况: Load={phys_load}, Speed={phys_speed}")
                    continue

                fn_x = f"{domain}_{load_id}_{speed_id}.npy"
                fn_s = f"{domain}_{load_id}_{speed_id}_S.npy"
                # ============================================

                path_x = os.path.join(Config.ATOMIC_DATA_DIR, fn_x)
                path_s = os.path.join(Config.ATOMIC_DATA_DIR, fn_s)

                if os.path.exists(path_x) and os.path.exists(path_s):
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
                else:
                    # 打印缺失的第一个文件路径，方便排查
                    if missing_files_count == 0:
                        print(f"[Error] Data Missing! Cannot find: {path_x}")
                        print(f"    (对应物理工况: Load={phys_load}, Speed={phys_speed})")
                    missing_files_count += 1

        if not xs:
            print(f"\n[Fatal Error] Dataset is empty!")
            print(f"   Expected {total_files_expected} files based on config.")
            print(f"   Found 0 files in {Config.ATOMIC_DATA_DIR}")
            print(f"   请检查 Config.LOAD_MAP/SPEED_MAP 是否与实际文件名一致！\n")
            return np.empty((0, Config.ENC_IN)), np.empty((0, 1)), [], []

        return np.concatenate(xs), np.concatenate(ss), np.concatenate(ys), np.concatenate(ls)

    def _add_noise(self, data, snr_db):
        """
        向声纹通道注入高斯白噪声
        """
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        return data + noise

    def __getitem__(self, idx):
        i = idx * self.stride
        x_win = self.x[i: i + self.window_size]
        s_win = self.s[i: i + self.window_size]
        y_win = self.y[i + self.predict_step: i + self.window_size + self.predict_step]

        # === 转速特征计算 (已修正) ===
        v_mean = np.mean(s_win)  # E[v]
        v2_mean = np.mean(s_win ** 2)  # E[v^2] - 代表平均动能

        # 归一化协变量
        scale = 3000.0
        cov = np.array([v_mean / scale, v2_mean / (scale ** 2)], dtype=np.float32)

        return torch.FloatTensor(x_win), torch.FloatTensor(y_win), torch.FloatTensor(cov), self.labels[i]

    def __len__(self):
        return self.n_samples
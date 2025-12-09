import os
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset
from config import Config


class MotorDataset(Dataset):
    # [修正] 加回 fault_types 参数，防止 run_evaluation 报错
    def __init__(self, atoms_list, mode='train', fault_types=None, noise_snr=None):
        self.window_size = Config.WINDOW_SIZE
        self.mode = mode
        self.noise_snr = noise_snr
        self.fault_types = fault_types

        # 1. 动态加载数据
        self.x, self.s, self.labels = self._load_data(atoms_list)

        # 2. [自动化] 训练模式下自动拟合 Scaler
        if mode == 'train':
            self._fit_scaler(self.x)

        # 3. 应用归一化
        self.x = self._apply_scaler(self.x)

        # 4. 读取配置中的预测长度
        self.pred_len = getattr(Config, 'PRED_LEN', 0)
        # 如果是预测模式，滑动步长可能需要调整，这里暂时保持一致

    def _load_data(self, atoms_list):
        xs, ss, ls = [], [], []

        # [修正] 恢复根据 fault_types 筛选数据的逻辑
        if self.mode == 'train':
            domains = ['HH']
        else:
            # 如果指定了故障类型，就只加载 HH + 指定故障
            # 否则加载所有 Config.DATA_DOMAINS 定义的类型
            target_faults = self.fault_types if self.fault_types else Config.TEST_FAULT_TYPES
            # 去重防止 HH 被加两次
            # 使用 sorted 确保加载顺序完全确定
            domains = sorted(list(set(['HH'] + target_faults)))

        for domain in domains:
            # 简单标签: HH=0, 其他=1
            label = 0 if domain == 'HH' else 1

            # 仅加载 atoms_list 中指定的工况
            for (load, speed) in atoms_list:
                # 查表获取 ID
                lid, sid = Config.LOAD_MAP.get(load), Config.SPEED_MAP.get(speed)
                if not lid or not sid: continue

                # 构建文件名
                fn_x = f"{domain}_{lid}_{sid}.npy"
                fn_s = f"{domain}_{lid}_{sid}_S.npy"
                path_x = os.path.join(Config.ATOMIC_DATA_DIR, fn_x)
                path_s = os.path.join(Config.ATOMIC_DATA_DIR, fn_s)

                if os.path.exists(path_x) and os.path.exists(path_s):
                    x_data = np.load(path_x)
                    s_data = np.load(path_s)  # Shape [L, 2]

                    # 测试时注入噪声
                    if self.mode == 'test' and self.noise_snr is not None:
                        x_data = self._add_noise(x_data, self.noise_snr)

                    xs.append(x_data)
                    ss.append(s_data)
                    ls.append(np.ones(len(x_data)) * label)

        if not xs:
            # 打印当前尝试加载的路径，方便排查路径问题
            print(f"[Warn] No data loaded for atoms: {atoms_list}")
            print(f"       Checked dir: {Config.ATOMIC_DATA_DIR}")
            return np.empty((0, Config.ENC_IN)), np.empty((0, 2)), []

        return np.concatenate(xs), np.concatenate(ss), np.concatenate(ls)

    def _fit_scaler(self, data):
        """计算并保存训练集的均值和方差"""
        if len(data) == 0: return
        print("[Scaler] Fitting global scaler on training data...")
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-6

        os.makedirs(os.path.dirname(Config.SCALER_PATH), exist_ok=True)
        with open(Config.SCALER_PATH, 'wb') as f:
            pickle.dump({'mean': mean, 'std': std}, f)

        self.mean = mean
        self.std = std
        print(f"[Scaler] Saved to {Config.SCALER_PATH}")

    def _apply_scaler(self, data):
        """应用归一化"""
        if len(data) == 0: return data

        if not hasattr(self, 'mean'):
            if os.path.exists(Config.SCALER_PATH):
                with open(Config.SCALER_PATH, 'rb') as f:
                    scaler = pickle.load(f)
                self.mean = scaler['mean']
                self.std = scaler['std']
            else:
                # 这是一个常见错误：如果没训练过直接跑测试，就没有 scaler
                # 为了防止代码崩在路径检查上，这里给个 warning 并返回原数据
                print(f"[Warn] Scaler not found at {Config.SCALER_PATH}. Using raw data.")
                return data

        return (data - self.mean) / self.std

    def _add_noise(self, data, snr_db):
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
        return data + noise

    def __getitem__(self, idx):
        stride = Config.WINDOW_SIZE if self.mode == 'train' else Config.WINDOW_SIZE
        i = idx * stride

        # 边界检查需要考虑预测长度
        total_len = self.window_size + (self.pred_len if self.pred_len > 0 else 0)
        if i + total_len > len(self.x):
            i = len(self.x) - total_len

        # 输入窗口 (History)
        x_win = self.x[i: i + self.window_size]
        s_win = self.s[i: i + self.window_size]

        # 标签窗口 (Target)
        if self.pred_len > 0:
            # [预测模式] 标签是紧接着输入之后的 P 个点
            y_win = self.x[i + self.window_size: i + self.window_size + self.pred_len]
        else:
            # [重构模式] 标签就是输入本身
            y_win = x_win

        # 协变量 (Speed)
        # 注意：预测任务中，协变量通常需要知道“未来的转速”或“当前的转速”
        # 这里为了简单且符合物理阻断逻辑，我们依然使用输入窗口的平均转速作为 Condition
        cov_win = np.mean(s_win, axis=0)
        norm_scale = np.array([3000.0, 9000000.0], dtype=np.float32)
        cov = cov_win / norm_scale

        return torch.FloatTensor(x_win), torch.FloatTensor(y_win), torch.FloatTensor(cov), self.labels[i]

    def __len__(self):
        # 长度计算也要考虑 pred_len
        total_len = self.window_size + (self.pred_len if self.pred_len > 0 else 0)
        stride = Config.WINDOW_SIZE
        if len(self.x) < total_len: return 0
        return (len(self.x) - total_len) // stride + 1
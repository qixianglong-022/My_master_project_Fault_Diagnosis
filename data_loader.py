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

        # 逻辑恢复：根据 mode 决定加载哪些故障
        if self.mode == 'train':
            domains = ['HH']
        else:
            target_faults = self.fault_types if self.fault_types else Config.TEST_FAULT_TYPES
            domains = sorted(list(set(['HH'] + target_faults)))

        for domain in domains:
            label = 0 if domain == 'HH' else 1
            for (load, speed) in atoms_list:
                lid, sid = Config.LOAD_MAP.get(load), Config.SPEED_MAP.get(speed)
                if not lid or not sid: continue

                fn_x = f"{domain}_{lid}_{sid}.npy"
                fn_s = f"{domain}_{lid}_{sid}_S.npy"
                path_x = os.path.join(Config.ATOMIC_DATA_DIR, fn_x)
                path_s = os.path.join(Config.ATOMIC_DATA_DIR, fn_s)

                if os.path.exists(path_x) and os.path.exists(path_s):
                    x_data = np.load(path_x)
                    s_data = np.load(path_s)

                    # === [核心热修复] ===
                    # 如果发现转速数据长度是特征数据的2倍，说明被压扁了，立刻 reshape 回来
                    if s_data.shape[0] == 2 * x_data.shape[0]:
                        # print(f"Auto-fixing shape for {fn_s}: {s_data.shape} -> ({x_data.shape[0]}, 2)")
                        s_data = s_data.reshape(-1, 2)
                    # ===================

                    # 长度对齐检查 (防止只有几帧的误差)
                    min_len = min(len(x_data), len(s_data))
                    x_data = x_data[:min_len]
                    s_data = s_data[:min_len]

                    if self.mode == 'test' and self.noise_snr is not None:
                        x_data = self._add_noise(x_data, self.noise_snr)

                    xs.append(x_data)
                    ss.append(s_data)
                    ls.append(np.ones(min_len) * label)

        if not xs:
            print(f"[Warn] No data loaded for atoms: {atoms_list}")
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
        # cov_win = np.mean(s_win, axis=0)
        # norm_scale = np.array([3000.0, 9000000.0], dtype=np.float32)
        # cov = cov_win / norm_scale

        # [修改后] 计算 Start, End, Mean
        v_start = s_win[0, 0]
        v_end = s_win[-1, 0]
        v_mean = np.mean(s_win[:, 0])
        v2_mean = np.mean(s_win[:, 1])

        # 归一化 (根据你的 3000 rpm)
        norm = 3000.0
        # 构造 3 维协变量: [Mean, Mean_Square, Slope_Indicator]
        # Slope_Indicator = (End - Start)
        cov = np.array([
            v_mean / norm,
            v2_mean / (norm ** 2),
            (v_end - v_start) / norm
        ], dtype=np.float32)

        return torch.FloatTensor(x_win), torch.FloatTensor(y_win), torch.FloatTensor(cov), self.labels[i]

    def __len__(self):
        # 长度计算也要考虑 pred_len
        total_len = self.window_size + (self.pred_len if self.pred_len > 0 else 0)
        stride = Config.WINDOW_SIZE
        if len(self.x) < total_len: return 0
        return (len(self.x) - total_len) // stride + 1
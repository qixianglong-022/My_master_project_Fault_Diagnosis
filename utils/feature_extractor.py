import numpy as np
from scipy.fftpack import dct


class FeatureExtractor:
    def __init__(self, config):
        """
        初始化特征提取器：支持 MFCC 与 LFCC 的并行提取
        """
        self.sr = config.SAMPLE_RATE
        self.frame_size = config.FRAME_SIZE
        self.hop_length = config.HOP_LENGTH
        self.n_fft = self.frame_size
        self.window = np.hanning(self.frame_size)

        # 1. 加载配置开关 (从 Config 中读取，提供默认值以防报错)
        self.use_mfcc = getattr(config, 'USE_MFCC', True)
        self.use_lfcc = getattr(config, 'USE_LFCC', False)

        self.n_mfcc = getattr(config, 'N_MFCC', 13)
        self.n_lfcc = getattr(config, 'N_LFCC', 13)

        # 滤波器数量 (通常 40 个滤波器足够)
        self.n_filters = 40

        # 2. 预计算滤波器组 (根据开关决定是否计算)
        self.mel_fbank = None
        self.linear_fbank = None

        if self.use_mfcc:
            self.mel_fbank = self._create_filter_bank(mode='mel')

        if self.use_lfcc:
            self.linear_fbank = self._create_filter_bank(mode='linear')

    def _create_filter_bank(self, mode):
        """
        内部工具函数：根据模式生成 Mel 或 Linear 滤波器组
        """
        low_freq = 0
        high_freq = self.sr / 2
        n_fft = self.n_fft
        n_filters = self.n_filters

        # A. 生成频点 (Frequency Points)
        if mode == 'linear':
            # === LFCC: 线性分布 (全频段均匀关注) ===
            points = np.linspace(low_freq, high_freq, n_filters + 2)
        else:
            # === MFCC: 梅尔分布 (低频密集，高频稀疏) ===
            low_mel = 2595 * np.log10(1 + low_freq / 700)
            high_mel = 2595 * np.log10(1 + high_freq / 700)
            mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
            points = 700 * (10 ** (mel_points / 2595) - 1)

        # B. 映射到 FFT Bin
        bin_points = np.floor((n_fft + 1) * points / self.sr).astype(int)

        # C. 构建三角滤波器矩阵
        fbank = np.zeros((n_filters, n_fft // 2 + 1))
        for i in range(1, n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            if center > left:
                for j in range(left, center):
                    fbank[i - 1, j] = (j - left) / (center - left)
            if right > center:
                for j in range(center, right):
                    fbank[i - 1, j] = (right - j) / (right - center)

        return fbank

    def _compute_stft(self, signal):
        """计算功率谱"""
        sig_len = len(signal)
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1
        if n_frames <= 0: return None

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        windowed_frames = frames * self.window
        mag_frames = np.abs(np.fft.rfft(windowed_frames, n=self.n_fft, axis=1))
        pow_frames = (mag_frames ** 2) / self.n_fft
        return pow_frames

    def extract_vib_features(self, signal):
        """提取振动特征 (RMS + Kurtosis)"""
        sig_len = len(signal)
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1
        if n_frames <= 0: return np.zeros((0, 2))

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        # RMS
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # Kurtosis (使用 eps 防止除零)
        mean = np.mean(frames, axis=1, keepdims=True)
        var = np.mean((frames - mean) ** 2, axis=1)
        m4 = np.mean((frames - mean) ** 4, axis=1)
        kurt = m4 / (var ** 2 + 1e-9)

        return np.stack([rms, kurt], axis=1)

    def extract_audio_features(self, signal):
        """
        提取声纹特征 (无需传入 config，直接使用 self.config)
        """
        # 1. 计算 STFT 功率谱
        pow_frames = self._compute_stft(signal)
        # 兜底返回：如果信号太短无法分帧，返回全0
        expected_dim = (self.n_mfcc if self.use_mfcc else 0) + (self.n_lfcc if self.use_lfcc else 0)
        if pow_frames is None:
            return np.zeros((1, expected_dim))

        features = []

        # 分支 A: MFCC
        if self.use_mfcc and self.mel_fbank is not None:
            mel_spec = np.dot(pow_frames, self.mel_fbank.T)
            mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)  # 数值稳定
            mel_spec = 10 * np.log10(mel_spec)
            # 取 1~N+1 丢弃第0个能量系数
            mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, 1:self.n_mfcc + 1]
            features.append(mfcc)

        # 分支 B: LFCC
        if self.use_lfcc and self.linear_fbank is not None:
            linear_spec = np.dot(pow_frames, self.linear_fbank.T)
            linear_spec = np.where(linear_spec == 0, np.finfo(float).eps, linear_spec)
            linear_spec = 10 * np.log10(linear_spec)
            lfcc = dct(linear_spec, type=2, axis=1, norm='ortho')[:, 1:self.n_lfcc + 1]
            features.append(lfcc)

        # 拼接
        if not features:
            return np.zeros((pow_frames.shape[0], 0))

        return np.concatenate(features, axis=1)
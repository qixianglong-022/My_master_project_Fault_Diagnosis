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
        self.n_mfcc = getattr(config, 'N_MFCC', 13)
        self.n_filters = 40
        self.mel_fbank = self._create_filter_bank()


    def _create_filter_bank(self):
        """
        内部工具函数：根据模式生成 Mel 或 Linear 滤波器组
        """
        low_freq = 0
        high_freq = self.sr / 2
        n_fft = self.n_fft
        n_filters = self.n_filters

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
        return pow_frames, mag_frames # 返回 mag 用于计算 SF

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
        [论文对齐] 提取 MFCC(13) + Spectral Flatness(1) + High/Low Ratio(1)
        """
        pow_frames, mag_frames = self._compute_stft(signal)
        if pow_frames is None:
            return np.zeros((1, self.n_mfcc + 2))

        # 1. MFCC
        mel_spec = np.dot(pow_frames, self.mel_fbank.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
        mel_spec = 10 * np.log10(mel_spec)
        mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, 1:self.n_mfcc + 1]

        # 2. Spectral Flatness (SF) = Geometric_Mean / Arithmetic_Mean
        # 加上微小量防止 log(0)
        mag_frames = mag_frames + 1e-10
        gmean = np.exp(np.mean(np.log(mag_frames), axis=1))
        amean = np.mean(mag_frames, axis=1)
        sf = (gmean / amean).reshape(-1, 1)

        # 3. Energy Ratio (High / Low)
        # 简单定义：Low < 1kHz, High > 1kHz
        # Index = Freq * N_FFT / SR
        split_idx = int(1000 * self.n_fft / self.sr)
        low_energy = np.sum(pow_frames[:, :split_idx], axis=1) + 1e-10
        high_energy = np.sum(pow_frames[:, split_idx:], axis=1) + 1e-10
        er = (high_energy / low_energy).reshape(-1, 1)
        # 取对数压缩范围，使其分布更友好
        er = np.log10(er)

        return np.concatenate([mfcc, sf, er], axis=1)
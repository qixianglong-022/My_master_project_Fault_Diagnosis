import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        """
        初始化特征提取器：预计算 FFT 窗函数、梅尔滤波器组和 DCT 矩阵，
        以实现 O(1) 级别的特征提取，避免重复计算。
        """
        self.sr = config.SAMPLE_RATE
        self.frame_size = config.FRAME_SIZE
        self.hop_length = config.HOP_LENGTH
        self.n_mfcc = config.N_MFCC

        # --- 1. 预计算汉宁窗 (Hanning Window) ---
        self.window = np.hanning(self.frame_size)

        # --- 2. 预计算梅尔滤波器组 (Mel Filter Bank) ---
        # 频率分辨率
        n_fft = self.frame_size
        self.n_fft = n_fft
        low_freq = 0
        high_freq = self.sr / 2
        n_mels = 40  # 梅尔滤波器数量，通常取 26-40

        # Mel 刻度转换
        mel_low = 2595 * np.log10(1 + low_freq / 700)
        mel_high = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        # 构建滤波器矩阵 [n_fft//2 + 1, n_mels]
        self.fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            left = int(bin_points[i - 1])
            center = int(bin_points[i])
            right = int(bin_points[i + 1])

            for j in range(left, center):
                self.fbank[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
            for j in range(center, right):
                self.fbank[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])

        # --- 3. 预计算 DCT 矩阵 (用于 MFCC) ---
        # [n_mels, n_mfcc] -> 这里我们需要 n_mfcc 个系数
        # 注意：librosa 默认包含 C0 (能量)，这里我们通常取 1:n_mfcc+1
        self.n_dct_filters = n_mels
        # 正交归一化 DCT 矩阵
        n = np.arange(n_mels)
        k = np.arange(self.n_mfcc + 1)[:, np.newaxis]  # 取 n_mfcc + 1 个，后面丢掉 C0
        self.dct_matrix = np.cos(np.pi / n_mels * (n + 0.5) * k)

    def _compute_stft(self, signal):
        """手动实现短时傅里叶变换 (STFT)"""
        # 1. 分帧 (Framing) - 使用 stride_tricks 实现零拷贝切片
        # 信号长度
        sig_len = len(signal)
        # 计算帧数
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1
        if n_frames <= 0:
            return None

        # 利用 numpy 的 stride tricks 快速分帧
        strides = signal.strides  # (bytes_per_sample,)
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        # 2. 加窗 (Windowing)
        windowed_frames = frames * self.window

        # 3. FFT -> Power Spectrum
        # rfft 只计算正频率部分，速度快一倍
        mag_frames = np.abs(np.fft.rfft(windowed_frames, n=self.n_fft, axis=1))
        pow_frames = (mag_frames ** 2) / self.n_fft

        return pow_frames

    def extract_vib_features(self, signal):
        """
        提取振动特征: RMS, Kurtosis
        signal: [Total_Len] (单通道)
        return: [Seq_Len, 2]
        """
        # 手动分帧逻辑
        sig_len = len(signal)
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)  # [N, Frame]

        # 1. RMS: sqrt(mean(x^2))
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        # 2. Kurtosis: E[(x-u)^4] / (E[(x-u)^2])^2
        mean = np.mean(frames, axis=1, keepdims=True)
        var = np.mean((frames - mean) ** 2, axis=1)
        m4 = np.mean((frames - mean) ** 4, axis=1)
        # 加上 1e-6 防止除零
        kurt = m4 / (var ** 2 + 1e-6)

        # 堆叠 [Seq, 2]
        return np.stack([rms, kurt], axis=1)

    def extract_audio_features(self, signal):
        """
        提取声纹特征: MFCC (Numpy version)
        signal: [Total_Len]
        return: [Seq_Len, n_mfcc]
        """
        # 1. 计算功率谱 [Seq_Len, n_fft/2+1]
        pow_frames = self._compute_stft(signal)
        if pow_frames is None: return np.zeros((1, self.n_mfcc))

        # 2. 通过 Mel 滤波器组 [Seq_Len, n_mels]
        # (N, F) dot (M, F).T -> (N, M)
        filter_banks = np.dot(pow_frames, self.fbank.T)

        # 3. 取对数 (数值稳定性: +1e-10)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 10 * np.log10(filter_banks)  # dB

        # 4. DCT 变换 [Seq_Len, n_mfcc+1]
        # (N, M) dot (K, M).T -> (N, K)
        mfcc = np.dot(filter_banks, self.dct_matrix.T)

        # 5. 丢弃 C0 (能量项)，保留 C1-C13
        return mfcc[:, 1:]

    def process_window(self, x_window):
        """
        总入口：处理一个多通道窗口
        x_window: [Window_Size, N_Channels]
                 假设顺序: [Vib1, Vib2, ..., Audio]
        config中定义了哪些是 Vib，哪些是 Audio
        """
        # 需要传入 Vib 和 Audio 的列索引范围
        # 这里为了简化，假设外部负责拆分好 Vib 和 Audio 的数据
        pass
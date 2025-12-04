import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        """
        初始化特征提取器：O(1) 级别的特征提取
        """
        self.sr = config.SAMPLE_RATE
        self.frame_size = config.FRAME_SIZE
        self.hop_length = config.HOP_LENGTH
        self.n_mfcc = config.N_MFCC

        # 1. 汉宁窗
        self.window = np.hanning(self.frame_size)

        # 2. 梅尔滤波器组
        n_fft = self.frame_size
        self.n_fft = n_fft
        low_freq = 0
        high_freq = self.sr / 2
        n_mels = 40

        mel_low = 2595 * np.log10(1 + low_freq / 700)
        mel_high = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft + 1) * hz_points / self.sr).astype(int)

        self.fbank = np.zeros((n_mels, n_fft // 2 + 1))
        for i in range(1, n_mels + 1):
            left = int(bin_points[i - 1])
            center = int(bin_points[i])
            right = int(bin_points[i + 1])

            for j in range(left, center):
                self.fbank[i - 1, j] = (j - bin_points[i - 1]) / (bin_points[i] - bin_points[i - 1])
            for j in range(center, right):
                self.fbank[i - 1, j] = (bin_points[i + 1] - j) / (bin_points[i + 1] - bin_points[i])

        # 3. DCT 矩阵
        self.n_dct_filters = n_mels
        n = np.arange(n_mels)
        k = np.arange(self.n_mfcc + 1)[:, np.newaxis]
        self.dct_matrix = np.cos(np.pi / n_mels * (n + 0.5) * k)

    def _compute_stft(self, signal):
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
        sig_len = len(signal)
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        mean = np.mean(frames, axis=1, keepdims=True)
        var = np.mean((frames - mean) ** 2, axis=1)
        m4 = np.mean((frames - mean) ** 4, axis=1)
        kurt = m4 / (var ** 2 + 1e-6)

        return np.stack([rms, kurt], axis=1)

    def extract_audio_features(self, signal):
        pow_frames = self._compute_stft(signal)
        if pow_frames is None: return np.zeros((1, self.n_mfcc))

        filter_banks = np.dot(pow_frames, self.fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 10 * np.log10(filter_banks)
        mfcc = np.dot(filter_banks, self.dct_matrix.T)

        return mfcc[:, 1:]
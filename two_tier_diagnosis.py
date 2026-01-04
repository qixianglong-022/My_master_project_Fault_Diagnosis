#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : two_tier_diagnosis.py
@Desc    : 两级诊断策略系统 - 哨兵模式(异常检测) + 专家模式(故障分类)
          部署版本：使用ONNX Runtime，适配树莓派4B
@Author  : Auto-generated
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from scipy.signal import decimate
from collections import deque

# 简化配置（不依赖完整Config类）
SAMPLE_RATE = 51200
FRAME_SIZE = 8192
HOP_LENGTH = 1024
RAW_WINDOW_SIZE = 51200
WINDOW_SIZE = (RAW_WINDOW_SIZE - FRAME_SIZE) // HOP_LENGTH + 1

# 特征维度
N_VIB_FEAT = 2  # RMS, Kurtosis
N_MFCC = 13
N_EXTRA_AUDIO = 2  # SF, ER
N_VIB_CHANNELS = 4
FEAT_DIM_VIB = N_VIB_CHANNELS * N_VIB_FEAT  # 8
FEAT_DIM_AUDIO = N_MFCC + N_EXTRA_AUDIO  # 15
ENC_IN = FEAT_DIM_VIB + FEAT_DIM_AUDIO  # 23

# 第四章配置
CH4_FREQ_DIM = 512
CH4_CURRENT_DIM = 128
CH4_AUDIO_DIM = 15
CH4_DECIMATION = 50
CH4_FFT_POINTS = 1024


class FeatureExtractor:
    """
    简化的特征提取器（不依赖Config类）
    """

    def __init__(self):
        self.sr = SAMPLE_RATE
        self.frame_size = FRAME_SIZE
        self.hop_length = HOP_LENGTH
        self.n_fft = self.frame_size
        self.window = np.hanning(self.frame_size)
        self.n_mfcc = N_MFCC
        self.n_filters = 40
        self.mel_fbank = self._create_filter_bank()

    def _create_filter_bank(self):
        """创建Mel滤波器组"""
        from scipy.fftpack import dct
        low_freq = 0
        high_freq = self.sr / 2
        n_fft = self.n_fft
        n_filters = self.n_filters

        low_mel = 2595 * np.log10(1 + low_freq / 700)
        high_mel = 2595 * np.log10(1 + high_freq / 700)
        mel_points = np.linspace(low_mel, high_mel, n_filters + 2)
        points = 700 * (10 ** (mel_points / 2595) - 1)

        bin_points = np.floor((n_fft + 1) * points / self.sr).astype(int)

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
        if n_frames <= 0:
            return None, None

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        windowed_frames = frames * self.window
        mag_frames = np.abs(np.fft.rfft(windowed_frames, n=self.n_fft, axis=1))
        pow_frames = (mag_frames ** 2) / self.n_fft
        return pow_frames, mag_frames

    def extract_vib_features(self, signal):
        """提取振动特征 (RMS + Kurtosis)"""
        sig_len = len(signal)
        n_frames = (sig_len - self.frame_size) // self.hop_length + 1
        if n_frames <= 0:
            return np.zeros((0, 2))

        strides = signal.strides
        shape = (n_frames, self.frame_size)
        strides = (strides[0] * self.hop_length, strides[0])
        frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

        rms = np.sqrt(np.mean(frames ** 2, axis=1))
        mean = np.mean(frames, axis=1, keepdims=True)
        var = np.mean((frames - mean) ** 2, axis=1)
        m4 = np.mean((frames - mean) ** 4, axis=1)
        kurt = m4 / (var ** 2 + 1e-9)

        return np.stack([rms, kurt], axis=1)

    def extract_audio_features(self, signal):
        """提取声纹特征 (MFCC + SF + ER)"""
        from scipy.fftpack import dct

        pow_frames, mag_frames = self._compute_stft(signal)
        if pow_frames is None:
            return np.zeros((1, self.n_mfcc + 2))

        # MFCC
        mel_spec = np.dot(pow_frames, self.mel_fbank.T)
        mel_spec = np.where(mel_spec == 0, np.finfo(float).eps, mel_spec)
        mel_spec = 10 * np.log10(mel_spec)
        mfcc = dct(mel_spec, type=2, axis=1, norm='ortho')[:, 1:self.n_mfcc + 1]

        # Spectral Flatness
        mag_frames = mag_frames + 1e-10
        gmean = np.exp(np.mean(np.log(mag_frames), axis=1))
        amean = np.mean(mag_frames, axis=1)
        sf = (gmean / amean).reshape(-1, 1)

        # Energy Ratio
        split_idx = int(1000 * self.n_fft / self.sr)
        low_energy = np.sum(pow_frames[:, :split_idx], axis=1) + 1e-10
        high_energy = np.sum(pow_frames[:, split_idx:], axis=1) + 1e-10
        er = np.log10(high_energy / low_energy).reshape(-1, 1)

        return np.concatenate([mfcc, sf, er], axis=1)


class SentinelPreprocessor:
    """
    哨兵模式特征工程（第三章）
    参考 preprocess_atomic.py
    """

    def __init__(self, scaler_path: Optional[str] = None):
        self.extractor = FeatureExtractor()

        # 加载标准化参数（已由模型目录提供，这里不需要）
        self.mean = None
        self.std = None

    def compute_rpm_from_square_wave(self, signal, fs=51200, pulses_per_rev=1):
        """从方波信号中提取瞬时转速"""
        threshold = (np.max(signal) + np.min(signal)) / 2
        binary_signal = (signal > threshold).astype(int)
        edges = np.diff(binary_signal)
        rising_indices = np.where(edges == 1)[0]

        if len(rising_indices) < 2:
            return 0.0

        intervals = np.diff(rising_indices)
        avg_interval = np.mean(intervals)
        rpm = (fs / avg_interval) * 60 / pulses_per_rev
        return rpm

    def process_window(self, vib_data: np.ndarray, audio_data: np.ndarray,
                       speed_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理一个窗口的数据，返回特征序列和协变量

        Args:
            vib_data: [N, 4] 振动数据（4个通道）
            audio_data: [N] 声纹数据
            speed_data: [N] 转速方波数据

        Returns:
            x_feat: [Seq_Len, Feat_Dim] 特征序列
            cov: [2] 协变量（速度均值和平方）
        """
        # 1. 提取振动特征（RMS + Kurtosis）
        vib_feats_list = []
        for i in range(vib_data.shape[1]):
            f = self.extractor.extract_vib_features(vib_data[:, i])
            vib_feats_list.append(f)
        vib_feats = np.concatenate(vib_feats_list, axis=1)  # [Seq_Len, 8]

        # 2. 提取声纹特征（MFCC）
        if audio_data.ndim == 1:
            audio_feats = self.extractor.extract_audio_features(audio_data)
        else:
            audio_feats_list = []
            for i in range(audio_data.shape[1]):
                f = self.extractor.extract_audio_features(audio_data[:, i])
                audio_feats_list.append(f)
            audio_feats = np.concatenate(audio_feats_list, axis=1)

        # 3. 对齐长度
        min_len = min(len(vib_feats), len(audio_feats))
        vib_feats = vib_feats[:min_len]
        audio_feats = audio_feats[:min_len]

        # 4. 拼接特征
        x_feat = np.concatenate([vib_feats, audio_feats], axis=1)  # [Seq_Len, 23]

        # 5. 计算转速协变量
        speed_down = []
        for i in range(min_len):
            s_idx = i * HOP_LENGTH
            e_idx = s_idx + FRAME_SIZE
            if e_idx > len(speed_data):
                break
            segment = speed_data[s_idx:e_idx]
            current_rpm = self.compute_rpm_from_square_wave(
                segment, fs=SAMPLE_RATE, pulses_per_rev=1
            )
            v_mean = current_rpm
            v2_mean = current_rpm ** 2
            speed_down.append([v_mean, v2_mean])

        # 再次对齐
        real_len = len(speed_down)
        x_feat = x_feat[:real_len]

        # 6. 标准化
        if self.mean is not None and self.std is not None:
            x_feat = (x_feat - self.mean) / self.std

        # 7. 计算协变量（窗口内的平均）
        if len(speed_down) > 0:
            speed_array = np.array(speed_down)
            v_bar = np.mean(speed_array[:, 0])
            v2_bar = np.mean(speed_array[:, 1])
            norm_scale = 3000.0
            cov = np.array([v_bar / norm_scale, v2_bar / (norm_scale ** 2)], dtype=np.float32)
        else:
            cov = np.array([0.0, 0.0], dtype=np.float32)

        return x_feat, cov


class ExpertPreprocessor:
    """
    专家模式特征工程（第四章）
    参考 preprocess_ch4_manager.py
    """

    def __init__(self, scaler_path: Optional[str] = None):
        self.original_fs = SAMPLE_RATE  # 51200
        self.decimation = CH4_DECIMATION
        self.target_fs = self.original_fs / self.decimation  # 1024 Hz
        self.fft_points = CH4_FFT_POINTS
        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63
        self.win_size_raw = self.fft_points * self.decimation  # 51200 (1秒)

        self.extractor = FeatureExtractor()

        # 加载标准化参数
        if scaler_path and os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            # 尝试自动查找（已由模型目录提供，这里不需要）
            print("[Warn] Ch4 scaler not found, using default")
            self.scaler = None

    def robust_compute_frequency_hz(self, signal, fs, default_hz=None):
        """鲁棒转速频率计算"""
        signal = np.convolve(signal, np.ones(5) / 5, mode='same')
        _min, _max = np.min(signal), np.max(signal)
        if _max - _min < 0.5:
            return float(default_hz) if default_hz else 0.0

        mid = (_max + _min) / 2
        high_th = mid + 0.2 * (_max - mid)
        low_th = mid - 0.2 * (_max - mid)

        state = 0
        rising_indices = []
        for i, val in enumerate(signal):
            if state == 0 and val > high_th:
                state = 1
                rising_indices.append(i)
            elif state == 1 and val < low_th:
                state = 0

        if len(rising_indices) < 2:
            return float(default_hz) if default_hz else 0.0

        intervals = np.diff(rising_indices)
        median_val = np.median(intervals)
        valid_intervals = intervals[np.abs(intervals - median_val) < 0.3 * median_val]
        avg_interval = np.mean(valid_intervals) if len(valid_intervals) > 0 else median_val

        return fs / avg_interval

    def _compute_fft(self, sig):
        """通用FFT计算"""
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))

        sig = sig - np.mean(sig)
        sig = sig[:self.fft_points] * self.hanning
        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction
        return spec[:self.fft_points // 2]

    def process_window(self, vib_data: np.ndarray, audio_data: np.ndarray,
                       speed_data: np.ndarray, current_data: np.ndarray,
                       default_hz: float = 15.0) -> Dict[str, np.ndarray]:
        """
        处理一个窗口的数据，返回多模态特征

        Args:
            vib_data: [N] 振动数据（主通道）
            audio_data: [N] 声纹数据
            speed_data: [N] 转速方波数据
            current_data: [N] 电流数据
            default_hz: 默认转速（Hz）

        Returns:
            dict: {
                'micro': [512] 低频频谱
                'macro': [512] 高频频谱
                'acoustic': [15] MFCC特征
                'current_spec': [128] 电流频谱
                'speed': float 转速(Hz)
                'load_proxy': float 归一化电流RMS
            }
        """
        # 确保数据长度足够
        if len(vib_data) < self.win_size_raw:
            # 补零
            pad_len = self.win_size_raw - len(vib_data)
            vib_data = np.pad(vib_data, (0, pad_len))
            speed_data = np.pad(speed_data, (0, pad_len))
            audio_data = np.pad(audio_data, (0, pad_len))
            current_data = np.pad(current_data, (0, pad_len))
        else:
            # 截取1秒数据
            vib_data = vib_data[:self.win_size_raw]
            speed_data = speed_data[:self.win_size_raw]
            audio_data = audio_data[:self.win_size_raw]
            current_data = current_data[:self.win_size_raw]

        # A. 计算物理转速
        hz = self.robust_compute_frequency_hz(speed_data, self.original_fs, default_hz)

        # B. Micro Stream (振动降采样 -> FFT)
        try:
            vib_low = decimate(vib_data, self.decimation, zero_phase=True)
            fft_micro = self._compute_fft(vib_low)
        except Exception:
            fft_micro = np.zeros(CH4_FREQ_DIM, dtype=np.float32)

        # C. Macro Stream (原始高频振动FFT)
        fft_macro = self._compute_fft(vib_data[:self.fft_points])

        # D. Current Stream (电流降采样 -> FFT)
        try:
            curr_low = decimate(current_data, self.decimation, zero_phase=True)
            fft_curr_full = self._compute_fft(curr_low)
            fft_curr = fft_curr_full[:CH4_CURRENT_DIM]
        except Exception:
            fft_curr = np.zeros(CH4_CURRENT_DIM, dtype=np.float32)

        # E. 负载RMS
        curr_rms = np.sqrt(np.mean(current_data ** 2))

        # F. Acoustic (MFCC)
        try:
            mfcc = np.mean(self.extractor.extract_audio_features(audio_data[:16384]), axis=0)
            if len(mfcc) != CH4_AUDIO_DIM:
                # 如果维度不匹配，截断或补零
                if len(mfcc) > CH4_AUDIO_DIM:
                    mfcc = mfcc[:CH4_AUDIO_DIM]
                else:
                    mfcc = np.pad(mfcc, (0, CH4_AUDIO_DIM - len(mfcc)))
        except Exception:
            mfcc = np.zeros(CH4_AUDIO_DIM, dtype=np.float32)

        # 标准化处理
        if self.scaler is not None:
            fft_micro = (fft_micro - self.scaler.get('micro_mean', 0)) / (self.scaler.get('micro_std', 1) + 1e-6)
            fft_macro = (fft_macro - self.scaler.get('macro_mean', 0)) / (self.scaler.get('macro_std', 1) + 1e-6)
            mfcc = (mfcc - self.scaler.get('ac_mean', 0)) / (self.scaler.get('ac_std', 1) + 1e-6)
            fft_curr = (fft_curr - self.scaler.get('curr_spec_mean', 0)) / (self.scaler.get('curr_spec_std', 1) + 1e-6)

            # 负载归一化
            curr_rms_ref = self.scaler.get('curr_rms_ref', 1.0)
            load_proxy = curr_rms / (curr_rms_ref + 1e-6)
            load_proxy = np.clip(load_proxy, 0.0, 3.0)
        else:
            load_proxy = curr_rms / 100.0  # 简单归一化
            load_proxy = np.clip(load_proxy, 0.0, 3.0)

        # 对数变换
        fft_micro = np.log1p(fft_micro)
        fft_macro = np.log1p(fft_macro)
        fft_curr = np.log1p(fft_curr)

        return {
            'micro': fft_micro.astype(np.float32),
            'macro': fft_macro.astype(np.float32),
            'acoustic': mfcc.astype(np.float32),
            'current_spec': fft_curr.astype(np.float32),
            'speed': np.array([hz], dtype=np.float32),
            'load_proxy': np.array([load_proxy], dtype=np.float32)
        }


class SentinelMode:
    """
    哨兵模式：RDLinear-AD异常检测
    刷新率：10 Hz
    使用ONNX Runtime
    """

    def __init__(self, model_dir: str):
        """
        初始化哨兵模式

        Args:
            model_dir: 模型目录（如 "ch3_RDLinear"），包含 model.onnx, scaler_params.pkl, threshold.npy
        """
        self.model_dir = Path(model_dir)
        self.preprocessor = SentinelPreprocessor()

        # 加载ONNX模型
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"[Sentinel] ONNX model loaded: {onnx_path}")
        print(f"    Input names: {self.input_names}")

        # 加载标准化参数
        scaler_path = self.model_dir / "scaler_params.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                self.preprocessor.mean = scaler['mean']
                self.preprocessor.std = scaler['std']
            print(f"[Sentinel] Scaler loaded: {scaler_path}")
        else:
            print("[Warn] Scaler not found, using default (no normalization)")
            self.preprocessor.mean = None
            self.preprocessor.std = None

        # 加载阈值
        threshold_path = self.model_dir / "threshold.npy"
        if threshold_path.exists():
            self.threshold = float(np.load(threshold_path))
        else:
            print("[Warn] Threshold not found, using default 1.0")
            self.threshold = 1.0

        print(f"[Sentinel] Threshold: {self.threshold:.4f}")

        # 缓冲区
        self.buffer_vib = []
        self.buffer_audio = []
        self.buffer_speed = []

        # 窗口参数
        self.window_size = RAW_WINDOW_SIZE  # 51200点
        self.hop_size = self.window_size // 10  # 10Hz刷新率

    def add_data(self, vib: np.ndarray, audio: np.ndarray, speed: np.ndarray):
        """添加数据到缓冲区"""
        self.buffer_vib.extend(vib)
        self.buffer_audio.extend(audio)
        self.buffer_speed.extend(speed)

    def process(self) -> Tuple[float, bool]:
        """
        处理缓冲区数据，返回SPE和是否异常

        Returns:
            spe: 重构误差
            is_anomaly: 是否异常
        """
        if len(self.buffer_vib) < self.window_size:
            return 0.0, False

        # 取出一个窗口
        vib_window = np.array(self.buffer_vib[:self.window_size])
        audio_window = np.array(self.buffer_audio[:self.window_size])
        speed_window = np.array(self.buffer_speed[:self.window_size])

        # 特征工程
        x_feat, cov = self.preprocessor.process_window(vib_window, audio_window, speed_window)

        if len(x_feat) == 0:
            return 0.0, False

        # 准备ONNX输入
        x_input = np.expand_dims(x_feat, axis=0).astype(np.float32)  # [1, Seq_Len, Feat_Dim]
        cov_input = np.expand_dims(cov, axis=0).astype(np.float32)  # [1, 2]

        # ONNX推理
        inputs = {
            self.input_names[0]: x_input,
            self.input_names[1]: cov_input
        }
        pred = self.session.run(None, inputs)[0]

        # 计算SPE
        diff = (x_input - pred) ** 2
        spe = np.mean(diff)

        # 判断是否异常
        is_anomaly = spe > self.threshold

        # 滑动窗口（移除hop_size个点）
        self.buffer_vib = self.buffer_vib[self.hop_size:]
        self.buffer_audio = self.buffer_audio[self.hop_size:]
        self.buffer_speed = self.buffer_speed[self.hop_size:]

        return spe, is_anomaly


class ExpertMode:
    """
    专家模式：Phys-RDLinear故障分类
    刷新率：1 Hz
    使用ONNX Runtime
    """

    def __init__(self, model_dir: str):
        """
        初始化专家模式

        Args:
            model_dir: 模型目录（如 "ch4_Phys-RDLinear"），包含 model.onnx, scaler_params.pkl
        """
        self.model_dir = Path(model_dir)
        self.preprocessor = ExpertPreprocessor()

        # 加载ONNX模型
        onnx_path = self.model_dir / "model.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        self.session = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        print(f"[Expert] ONNX model loaded: {onnx_path}")
        print(f"    Input names: {self.input_names}")

        # 加载标准化参数
        scaler_path = self.model_dir / "scaler_params.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.preprocessor.scaler = pickle.load(f)
            print(f"[Expert] Scaler loaded: {scaler_path}")
        else:
            print("[Warn] Scaler not found, using default")
            self.preprocessor.scaler = None

        # 类别名称
        self.class_names = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']

        # 缓冲区
        self.buffer_vib = []
        self.buffer_audio = []
        self.buffer_speed = []
        self.buffer_current = []

        # 窗口参数
        self.window_size = 51200  # 1秒数据
        self.hop_size = self.window_size  # 1Hz刷新率

    def add_data(self, vib: np.ndarray, audio: np.ndarray, speed: np.ndarray, current: np.ndarray):
        """添加数据到缓冲区"""
        self.buffer_vib.extend(vib)
        self.buffer_audio.extend(audio)
        self.buffer_speed.extend(speed)
        self.buffer_current.extend(current)

    def process(self) -> Tuple[int, str, float]:
        """
        处理缓冲区数据，返回故障类别

        Returns:
            class_id: 故障类别ID (0-7)
            class_name: 故障类别名称
            confidence: 置信度
        """
        if len(self.buffer_vib) < self.window_size:
            return -1, "Unknown", 0.0

        # 取出一个窗口
        vib_window = np.array(self.buffer_vib[:self.window_size])
        audio_window = np.array(self.buffer_audio[:self.window_size])
        speed_window = np.array(self.buffer_speed[:self.window_size])
        current_window = np.array(self.buffer_current[:self.window_size])

        # 特征工程
        features = self.preprocessor.process_window(
            vib_window, audio_window, speed_window, current_window
        )

        # 准备ONNX输入
        micro = features['micro'].reshape(1, CH4_FREQ_DIM, 1).astype(np.float32)
        macro = features['macro'].reshape(1, CH4_FREQ_DIM, 1).astype(np.float32)
        acoustic = features['acoustic'].reshape(1, CH4_AUDIO_DIM).astype(np.float32)
        current_spec = features['current_spec'].reshape(1, CH4_CURRENT_DIM).astype(np.float32)
        speed = features['speed'].reshape(1, 1).astype(np.float32)
        load_proxy = features['load_proxy'].reshape(1, 1).astype(np.float32)

        # ONNX推理
        inputs = {
            self.input_names[0]: micro,
            self.input_names[1]: macro,
            self.input_names[2]: acoustic,
            self.input_names[3]: current_spec,
            self.input_names[4]: speed,
            self.input_names[5]: load_proxy
        }
        logits = self.session.run(None, inputs)[0]

        # 计算概率和类别
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        class_id = int(np.argmax(probs, axis=1)[0])
        confidence = float(probs[0, class_id])

        # 滑动窗口
        self.buffer_vib = self.buffer_vib[self.hop_size:]
        self.buffer_audio = self.buffer_audio[self.hop_size:]
        self.buffer_speed = self.buffer_speed[self.hop_size:]
        self.buffer_current = self.buffer_current[self.hop_size:]

        return class_id, self.class_names[class_id], confidence


class TwoTierDiagnosisSystem:
    """
    两级诊断系统
    适配树莓派部署，使用ONNX Runtime
    """

    def __init__(self,
                 sentinel_model_dir: str,
                 expert_model_dir: str):
        """
        初始化系统

        Args:
            sentinel_model_dir: 哨兵模式模型目录（如 "ch3_RDLinear"）
            expert_model_dir: 专家模式模型目录（如 "ch4_Phys-RDLinear"）
        """
        # 初始化模式
        self.sentinel = SentinelMode(sentinel_model_dir)
        self.expert = ExpertMode(expert_model_dir)

        # 系统状态
        self.current_mode = "sentinel"  # "sentinel" or "expert"
        self.cooldown_until = 0.0  # 冷却期结束时间
        self.cooldown_duration = 3.0  # 冷却期3秒

        # 统计信息
        self.stats = {
            'sentinel_calls': 0,
            'expert_calls': 0,
            'anomalies_detected': 0,
            'faults_diagnosed': 0
        }

        print(f"\n{'=' * 60}")
        print(f">>> Two-Tier Diagnosis System Initialized")
        print(f"    Sentinel Mode: 10 Hz")
        print(f"    Expert Mode: 1 Hz")
        print(f"    Cooldown: {self.cooldown_duration}s")
        print(f"{'=' * 60}\n")

    def read_txt_file(self, txt_path: str) -> Optional[np.ndarray]:
        """
        读取TXT文件（参考preprocess_atomic.py和preprocess_ch4_manager.py）

        Args:
            txt_path: TXT文件绝对路径

        Returns:
            data: [N, M] 数据数组，None表示读取失败
        """
        data_start_line = None

        try:
            # 预扫描：寻找数据起始行
            with open(txt_path, 'rb') as f:
                lines = f.readlines()
                for i, line in enumerate(lines[:50]):
                    try:
                        line_str = line.decode('gb18030', errors='ignore')
                    except:
                        continue
                    if 'Time' in line_str and 'Data Channels' in line_str:
                        data_start_line = i + 1
                        break
        except Exception as e:
            print(f"[Warn] 预扫描文件失败 {txt_path}: {e}")
            return None

        if data_start_line is None:
            data_start_line = 17  # 默认值

        # 读取数据
        try:
            df = pd.read_csv(
                txt_path,
                sep='\t',
                encoding='gb18030',
                header=None,
                skiprows=data_start_line,
                engine='python'
            )
            return df.values
        except Exception as e:
            try:
                # 尝试GBK编码
                df = pd.read_csv(
                    txt_path,
                    sep='\t',
                    encoding='gbk',
                    header=None,
                    skiprows=data_start_line,
                    engine='python'
                )
                return df.values
            except Exception as e2:
                print(f"[Error] 读取文件失败 {txt_path}: {e2}")
                return None

    def extract_channels(self, data: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        从数据中提取所需通道（参考config.py）

        Args:
            data: [N, M] 原始数据数组

        Returns:
            vib_data: [N, 4] 振动数据（4个通道：8, 10, 11, 12）
            audio_data: [N] 声纹数据（通道20）
            speed_data: [N] 转速数据（通道1）
            current_data: [N] 电流数据（通道6）
        """
        max_col = data.shape[1]

        # 通道索引（参考config.py）
        COL_INDEX_SPEED = 1
        COL_INDICES_VIB = [8, 10, 11, 12]
        COL_INDICES_AUDIO = [20]
        COL_INDEX_CURRENT = 6

        # 检查列索引是否有效
        required_cols = [COL_INDEX_SPEED] + COL_INDICES_VIB + COL_INDICES_AUDIO + [COL_INDEX_CURRENT]
        if max(required_cols) >= max_col:
            print(f"[Warn] 数据列数不足: 需要至少{max(required_cols) + 1}列，实际{max_col}列")
            # 使用可用列，缺失的用零填充
            vib_data = np.zeros((data.shape[0], len(COL_INDICES_VIB)), dtype=np.float32)
            for i, col_idx in enumerate(COL_INDICES_VIB):
                if col_idx < max_col:
                    vib_data[:, i] = data[:, col_idx].astype(np.float32)

            audio_data = np.zeros(data.shape[0], dtype=np.float32)
            if COL_INDICES_AUDIO[0] < max_col:
                audio_data = data[:, COL_INDICES_AUDIO[0]].astype(np.float32)

            speed_data = np.zeros(data.shape[0], dtype=np.float32)
            if COL_INDEX_SPEED < max_col:
                speed_data = data[:, COL_INDEX_SPEED].astype(np.float32)

            current_data = np.zeros(data.shape[0], dtype=np.float32)
            if COL_INDEX_CURRENT < max_col:
                current_data = data[:, COL_INDEX_CURRENT].astype(np.float32)
        else:
            # 正常提取
            vib_data = data[:, COL_INDICES_VIB].astype(np.float32)  # [N, 4]
            audio_data = data[:, COL_INDICES_AUDIO[0]].astype(np.float32)  # [N]
            speed_data = data[:, COL_INDEX_SPEED].astype(np.float32)  # [N]
            current_data = data[:, COL_INDEX_CURRENT].astype(np.float32)  # [N]

        return vib_data, audio_data, speed_data, current_data

    def process_txt_files(self, txt_paths: list, output_path: Optional[str] = None):
        """
        处理多个TXT文件

        Args:
            txt_paths: TXT文件路径列表（绝对路径）
            output_path: 输出结果文件路径（可选）
        """
        print(f">>> Processing {len(txt_paths)} TXT files...")

        # 读取并合并所有TXT文件的数据
        all_vib_data = []
        all_audio_data = []
        all_speed_data = []
        all_current_data = []

        for txt_path in txt_paths:
            print(f"    Reading: {os.path.basename(txt_path)}")

            # 读取TXT文件
            data = self.read_txt_file(txt_path)
            if data is None:
                print(f"    [Skip] Failed to read {txt_path}")
                continue

            # 提取通道
            vib_data, audio_data, speed_data, current_data = self.extract_channels(data)

            all_vib_data.append(vib_data)
            all_audio_data.append(audio_data)
            all_speed_data.append(speed_data)
            all_current_data.append(current_data)

            print(
                f"        Shape: {data.shape} -> Vib={vib_data.shape}, Audio={audio_data.shape}, Speed={speed_data.shape}, Current={current_data.shape}")

        if len(all_vib_data) == 0:
            print("[Error] No valid data loaded")
            return

        # 合并所有文件的数据
        vib_data = np.concatenate(all_vib_data, axis=0)
        audio_data = np.concatenate(all_audio_data, axis=0)
        speed_data = np.concatenate(all_speed_data, axis=0)
        current_data = np.concatenate(all_current_data, axis=0)

        print(
            f"\n    Total data shape: Vib={vib_data.shape}, Audio={audio_data.shape}, Speed={speed_data.shape}, Current={current_data.shape}")

        # 结果记录
        results = []

        # 模拟实时处理（按采样率处理）
        sample_rate = SAMPLE_RATE  # 51200 Hz
        chunk_size = sample_rate // 10  # 每0.1秒处理一次（10Hz）

        total_samples = len(vib_data)
        processed = 0

        print(f"\n>>> Starting real-time processing...")
        print(f"    Chunk size: {chunk_size} samples (0.1s)")
        print(f"    Total samples: {total_samples}\n")

        while processed < total_samples:
            # 计算当前时间
            current_time = time.time()

            # 检查冷却期
            if current_time < self.cooldown_until:
                # 冷却期内，强制使用哨兵模式
                self.current_mode = "sentinel"

            # 获取数据块
            end_idx = min(processed + chunk_size, total_samples)
            vib_chunk = vib_data[processed:end_idx]
            audio_chunk = audio_data[processed:end_idx]
            speed_chunk = speed_data[processed:end_idx]
            current_chunk = current_data[processed:end_idx]

            # 根据模式处理
            if self.current_mode == "sentinel":
                # 哨兵模式
                self.sentinel.add_data(vib_chunk, audio_chunk, speed_chunk)
                spe, is_anomaly = self.sentinel.process()
                self.stats['sentinel_calls'] += 1

                if is_anomaly:
                    self.stats['anomalies_detected'] += 1
                    print(f"[Sentinel] Anomaly detected! SPE={spe:.4f} > {self.sentinel.threshold:.4f}")
                    # 切换到专家模式
                    self.current_mode = "expert"
                    # 清空专家模式缓冲区，准备接收新数据
                    self.expert.buffer_vib = []
                    self.expert.buffer_audio = []
                    self.expert.buffer_speed = []
                    self.expert.buffer_current = []

                results.append({
                    'time': processed / sample_rate,
                    'mode': 'sentinel',
                    'spe': spe,
                    'is_anomaly': is_anomaly,
                    'fault_class': None,
                    'confidence': None
                })

            else:
                # 专家模式
                self.expert.add_data(vib_chunk, audio_chunk, speed_chunk, current_chunk)
                class_id, class_name, confidence = self.expert.process()
                self.stats['expert_calls'] += 1

                if class_id >= 0:
                    self.stats['faults_diagnosed'] += 1
                    print(f"[Expert] Fault diagnosed: {class_name} (Confidence: {confidence:.2%})")

                    results.append({
                        'time': processed / sample_rate,
                        'mode': 'expert',
                        'spe': None,
                        'is_anomaly': None,
                        'fault_class': class_name,
                        'confidence': confidence
                    })

                    # 诊断完成，返回哨兵模式并设置冷却期
                    self.current_mode = "sentinel"
                    self.cooldown_until = current_time + self.cooldown_duration
                    print(f"[System] Returning to Sentinel Mode (Cooldown: {self.cooldown_duration}s)")

            processed = end_idx

            # 显示进度
            if processed % (sample_rate * 5) == 0:  # 每5秒显示一次
                progress = processed / total_samples * 100
                print(f"    Progress: {progress:.1f}% ({processed}/{total_samples} samples)")

        # 保存结果
        if output_path:
            results_df = pd.DataFrame(results)
            results_df.to_csv(output_path, index=False)
            print(f"\n>>> Results saved to: {output_path}")

        # 打印统计信息
        print(f"\n{'=' * 60}")
        print(f">>> Processing Complete")
        print(f"    Sentinel calls: {self.stats['sentinel_calls']}")
        print(f"    Expert calls: {self.stats['expert_calls']}")
        print(f"    Anomalies detected: {self.stats['anomalies_detected']}")
        print(f"    Faults diagnosed: {self.stats['faults_diagnosed']}")
        print(f"{'=' * 60}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Two-Tier Diagnosis System (Raspberry Pi Deployment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (single TXT file)
  python two_tier_diagnosis.py \\
    --input "C:\\毕业材料_齐祥龙\\电机故障数据集\\实验台数据采集\\第2组——电机健康状态：转子不平衡（RU）\\RU-4-2.txt" \\
    --sentinel-dir ch3_RDLinear \\
    --expert-dir ch4_Phys-RDLinear \\
    --output results.csv

  # Multiple TXT files
  python two_tier_diagnosis.py \\
    --input "path1.txt" "path2.txt" "path3.txt" \\
    --sentinel-dir ch3_RDLinear \\
    --expert-dir ch4_Phys-RDLinear \\
    --output results.csv

Note:
  Model directories should contain:
    - model.onnx (ONNX model file)
    - scaler_params.pkl (normalization parameters)
    - threshold.npy (for Chapter 3 models only)
        """
    )

    parser.add_argument('--input', '-i', type=str, required=True, nargs='+',
                        help='Input TXT file path(s) (absolute paths, can specify multiple files)')
    parser.add_argument('--sentinel-dir', '-s', type=str, required=True,
                        help='Sentinel mode model directory (e.g., ch3_RDLinear)')
    parser.add_argument('--expert-dir', '-e', type=str, required=True,
                        help='Expert mode model directory (e.g., ch4_Phys-RDLinear)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output CSV file path')

    args = parser.parse_args()

    # 验证输入文件
    txt_paths = args.input if isinstance(args.input, list) else [args.input]
    valid_paths = []
    for txt_path in txt_paths:
        if os.path.exists(txt_path):
            valid_paths.append(txt_path)
        else:
            print(f"[Warn] Input file not found: {txt_path}")

    if len(valid_paths) == 0:
        print(f"[Error] No valid input files found")
        sys.exit(1)

    # 验证模型目录
    if not os.path.exists(args.sentinel_dir):
        print(f"[Error] Sentinel model directory not found: {args.sentinel_dir}")
        sys.exit(1)

    if not os.path.exists(args.expert_dir):
        print(f"[Error] Expert model directory not found: {args.expert_dir}")
        sys.exit(1)

    # 初始化系统
    try:
        system = TwoTierDiagnosisSystem(
            sentinel_model_dir=args.sentinel_dir,
            expert_model_dir=args.expert_dir
        )
    except Exception as e:
        print(f"[Error] Failed to initialize system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 处理TXT文件
    try:
        system.process_txt_files(valid_paths, args.output)
    except Exception as e:
        print(f"[Error] Processing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
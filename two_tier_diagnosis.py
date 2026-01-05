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
import psutil  # 新增: 用于监控CPU和内存
import csv
import datetime

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

            # --- [Fix Start] ---
            # 1. 计算均值 (Feature 1 & 2)
            v_bar = np.mean(speed_array[:, 0])
            v2_bar = np.mean(speed_array[:, 1])

            # 2. 计算斜率 (Feature 3: End - Start)
            # speed_array[:, 0] 是瞬时转速序列
            v_start = speed_array[0, 0]
            v_end = speed_array[-1, 0]
            v_slope = v_end - v_start

            norm_scale = 3000.0

            # 3. 构造 3维 向量 [Mean, Mean_Square, Slope]
            cov = np.array([
                v_bar / norm_scale,
                v2_bar / (norm_scale ** 2),
                v_slope / norm_scale  # 补上缺失的第3个特征
            ], dtype=np.float32)
            # --- [Fix End] ---

        else:
            # 兜底逻辑也要改成 3 维
            cov = np.array([0.0, 0.0, 0.0], dtype=np.float32)

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

        # 专家模式模型是单通道输入的，如果传入了多通道(N, 4)，只取第1个通道(电机振动)
        if vib_data.ndim > 1:
            vib_data = vib_data[:, 0]

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

        # ---------------------------------------------------------------------
        # [Fix] 1. 先进行对数变换 (Log Transform first)
        #       这一步是为了压缩数据的动态范围，处理长尾分布
        # ---------------------------------------------------------------------
        fft_micro = np.log1p(fft_micro)
        fft_macro = np.log1p(fft_macro)
        fft_curr = np.log1p(fft_curr)
        # 注意：mfcc 在提取时已经做过 log10，这里不需要再次 log

        # ---------------------------------------------------------------------
        # [Fix] 2. 后进行标准化 (Standardization second)
        #       确保输入模型的特征符合标准正态分布 (0 mean, 1 std)
        # ---------------------------------------------------------------------
        if self.scaler is not None:
            # 使用 scaler 中的参数 (这些参数应该是基于 log 后的数据计算出来的)
            fft_micro = (fft_micro - self.scaler.get('micro_mean', 0)) / (self.scaler.get('micro_std', 1) + 1e-6)
            fft_macro = (fft_macro - self.scaler.get('macro_mean', 0)) / (self.scaler.get('macro_std', 1) + 1e-6)

            # MFCC 本身就可以直接标准化
            mfcc = (mfcc - self.scaler.get('ac_mean', 0)) / (self.scaler.get('ac_std', 1) + 1e-6)

            fft_curr = (fft_curr - self.scaler.get('curr_spec_mean', 0)) / (
                        self.scaler.get('curr_spec_std', 1) + 1e-6)

            # 负载归一化
            curr_rms_ref = self.scaler.get('curr_rms_ref', 1.0)
            load_proxy = curr_rms / (curr_rms_ref + 1e-6)
            load_proxy = np.clip(load_proxy, 0.0, 3.0)
        else:
            # 兜底逻辑
            load_proxy = curr_rms / 100.0
            load_proxy = np.clip(load_proxy, 0.0, 3.0)

        # ---------------------------------------------------------------------
        # 返回结果
        # ---------------------------------------------------------------------
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

    def __init__(self, model_dir: str, adaptive: bool = True):
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
            if adaptive:
                # [策略] 简单的工况适配策略
                # 如果你知道这是 400kg 工况，通常 SPE 会比 200kg 高 2-4 倍
                # 这里给一个经验性的放宽系数，或者你可以像 run_evaluation 那样读取一小段数据来校准
                self.threshold = self.threshold * 5.0
                print(f"[Sentinel] Adaptive Threshold Enabled: {self.threshold:.4f} -> {self.threshold:.4f}")
            else:
                self.threshold = self.threshold
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

        # 在 SentinelMode 的 __init__ 中添加打印
        print(f"DEBUG: Input 'cov' shape requirement: {self.session.get_inputs()[1].shape}")

        # [新增] 读取模型元数据 (参数量 & FLOPs)
        summary_path = self.model_dir / "export_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                info = json.load(f).get('model_info', {})
                self.params = info.get('params', 'Unknown')
                self.flops = info.get('flops', 'Unknown')
                print(f"[Sentinel] Model Info: Params={self.params}, FLOPs={self.flops}")
        else:
            self.params = "Unknown"
            print("[Warn] Sentinel export_summary.json not found.")

    def add_data(self, vib: np.ndarray, audio: np.ndarray, speed: np.ndarray):
        """添加数据到缓冲区"""
        self.buffer_vib.extend(vib)
        self.buffer_audio.extend(audio)
        self.buffer_speed.extend(speed)

    def process(self) -> Tuple[float, bool, float]:
        """
        处理缓冲区数据，返回SPE和是否异常

        Returns:
            spe: 重构误差
            is_anomaly: 是否异常
        """
        if len(self.buffer_vib) < self.window_size:
            return 0.0, False, 0.0

        # 取出一个窗口
        vib_window = np.array(self.buffer_vib[:self.window_size])
        audio_window = np.array(self.buffer_audio[:self.window_size])
        speed_window = np.array(self.buffer_speed[:self.window_size])

        # 特征工程
        x_feat, cov = self.preprocessor.process_window(vib_window, audio_window, speed_window)

        if len(x_feat) == 0:
            return 0.0, False

        # 准备ONNX输入
        x_input = np.expand_dims(x_feat, axis=0).astype(np.float32)
        # cov 已经是 [3] 维，expand 后变成 [1, 3]，完全符合模型要求的 ['batch', 3]
        cov_input = np.expand_dims(cov, axis=0).astype(np.float32)

        # 计时开始
        t0 = time.perf_counter()

        # ONNX推理
        inputs = {
            self.input_names[0]: x_input,
            self.input_names[1]: cov_input
        }
        pred = self.session.run(None, inputs)[0]

        # 计时结束 (毫秒)
        latency_ms = (time.perf_counter() - t0) * 1000

        # 计算SPE
        diff = (x_input - pred) ** 2
        spe = np.mean(diff)

        # 判断是否异常
        is_anomaly = spe > self.threshold

        # 滑动窗口（移除hop_size个点）
        self.buffer_vib = self.buffer_vib[self.hop_size:]
        self.buffer_audio = self.buffer_audio[self.hop_size:]
        self.buffer_speed = self.buffer_speed[self.hop_size:]

        return spe, is_anomaly, latency_ms


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

        # 读取模型元数据
        summary_path = self.model_dir / "export_summary.json"
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                info = json.load(f).get('model_info', {})
                self.params = info.get('params', 'Unknown')
                self.flops = info.get('flops', 'Unknown')
                print(f"[Expert] Model Info: Params={self.params}, FLOPs={self.flops}")
        else:
            self.params = "Unknown"

    def add_data(self, vib: np.ndarray, audio: np.ndarray, speed: np.ndarray, current: np.ndarray):
        """添加数据到缓冲区"""
        self.buffer_vib.extend(vib)
        self.buffer_audio.extend(audio)
        self.buffer_speed.extend(speed)
        self.buffer_current.extend(current)

    def process(self) -> Tuple[int, str, float, float]:  # 返回值增加 latency
        if len(self.buffer_vib) < self.window_size:
            return -1, "Unknown", 0.0, 0.0
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

        # 计时开始
        t0 = time.perf_counter()

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

        # 计时结束 (毫秒)
        latency_ms = (time.perf_counter() - t0) * 1000

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

        return class_id, self.class_names[class_id], confidence, latency_ms


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


    def detect_header_line(self, txt_path: str) -> int:
        """
        [New] 仅扫描文件头部以获取数据起始行，不读取整个文件
        """
        data_start_line = 17  # 默认值
        try:
            with open(txt_path, 'rb') as f:
                # 只读前100行进行探测
                for i in range(100):
                    line = f.readline()
                    try:
                        line_str = line.decode('gb18030', errors='ignore')
                    except:
                        continue
                    if 'Time' in line_str and 'Data Channels' in line_str:
                        data_start_line = i + 1
                        break
        except Exception as e:
            print(f"[Warn] Header detection failed for {txt_path}: {e}, using default {data_start_line}")

        return data_start_line

    def process_txt_files(self, txt_paths: list, output_path: Optional[str] = None):
        """
        [Modified] 流式处理多个TXT文件，极低内存占用
        """
        print(f">>> Processing {len(txt_paths)} TXT files (Streaming Mode)...")

        # 兜底逻辑：如果未指定输出路径，自动生成默认文件名
        if output_path is None:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"diagnosis_results_{timestamp_str}.csv"
            print(f"    [Warn] No output path specified. Using default: {output_path}")

        # 定义 CSV 表头
        csv_headers = [
            'Timestamp',
            'Mode',
            'Sentinel_Result',
            'Expert_Diagnosis',
            'Confidence',
            'SPE',
            'Sentinel_Time_ms',  # 哨兵推理时间
            'Expert_Time_ms',  # 专家推理时间
            'CPU_Usage_%'  # 当前CPU使用率
        ]

        # 初始化 CSV 文件
        if output_path:
            try:
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(csv_headers)
                print(f"    [IO] CSV initialized at: {output_path}")
            except Exception as e:
                print(f"    [Error] Failed to init CSV: {e}")
                output_path = None

        # 实时模拟参数
        sim_step_size = 5120  # 每次推进0.1秒 (对应哨兵模式的刷新率)
        disk_chunk_size = 51200  # 每次从磁盘读取1秒数据 (减少IO次数)
        total_processed_samples = 0
        results = []

        # 状态变量：记录触发专家模式时的那个 SPE 值
        triggering_spe = "N/A"

        # CPU 监控
        process_monitor = psutil.Process()
        process_monitor.cpu_percent(interval=None)

        print(f"\n>>> Starting streaming processing...")
        print(f"    Disk Chunk: {disk_chunk_size} samples")
        print(f"    Sim Step  : {sim_step_size} samples (0.1s system tick)")

        start_time_wall = time.time()

        for txt_path in txt_paths:
            print(f"\n    [Stream] Opening: {os.path.basename(txt_path)}")

            # 1. 探测起始行
            skip_rows = self.detect_header_line(txt_path)

            # 2. 创建流式读取器 (Iterator)
            try:
                # 注意：这里使用了 chunksize，pandas 不会一次性读取文件
                reader = pd.read_csv(
                    txt_path,
                    sep='\t',
                    encoding='gb18030',  # 或 gbk
                    header=None,
                    skiprows=skip_rows,
                    engine='c',  # c 引擎更快，内存更少
                    chunksize=disk_chunk_size,
                    on_bad_lines='skip'
                )
            except Exception as e:
                print(f"    [Error] Failed to open stream {txt_path}: {e}")
                continue

            # 3. 逐块读取磁盘数据
            for chunk_df in reader:
                # 提取数据 (此时 chunk_df 只有 disk_chunk_size 行，内存很小)
                # 注意：extract_channels 需要适配 DataFrame 或 numpy
                raw_data = chunk_df.values
                vib_chunk_large, audio_chunk_large, speed_chunk_large, current_chunk_large = self.extract_channels(
                    raw_data)

                # 4. 将大块数据切分为微小的“时间片”来喂给系统
                # 这是为了模拟真实的时间流逝，保证能及时切换模式
                num_steps = (len(vib_chunk_large) + sim_step_size - 1) // sim_step_size

                for i in range(num_steps):
                    s_idx = i * sim_step_size
                    e_idx = min(s_idx + sim_step_size, len(vib_chunk_large))

                    if s_idx >= len(vib_chunk_large):
                        break

                    # 获取当前的微批次数据 (0.1秒)
                    vib_step = vib_chunk_large[s_idx:e_idx]
                    audio_step = audio_chunk_large[s_idx:e_idx]
                    speed_step = speed_chunk_large[s_idx:e_idx]
                    current_step = current_chunk_large[s_idx:e_idx]

                    current_sim_time = total_processed_samples / SAMPLE_RATE

                    # 检查冷却时间
                    if hasattr(self, 'cooldown_until_sim_time') and current_sim_time < self.cooldown_until_sim_time:
                        self.current_mode = "sentinel"

                    row_data = None

                    # 根据模式处理
                    if self.current_mode == "sentinel":
                        self.sentinel.add_data(vib_step, audio_step, speed_step)

                        # 循环调用 process 直到缓冲区不足 (应对边界情况)
                        while True:
                            process_monitor.cpu_percent(interval=None) # 1. 测量前重置计数器 (忽略非计算时间的消耗)

                            spe, is_anomaly, lat = self.sentinel.process()

                            current_cpu = process_monitor.cpu_percent(interval=None) # 2. 测量后立即获取，得到处理期间的 CPU 占用

                            if lat == 0.0:  # 没有发生处理(数据不够)
                                break

                            self.stats['sentinel_calls'] += 1

                            sentinel_res = "Anomaly" if is_anomaly else "Normal"
                            current_spe_str = f"{spe:.4f}"

                            # 如果发现异常，记录这个 SPE，以便稍后传给专家模式
                            if is_anomaly:
                                triggering_spe = current_spe_str

                            # 准备 CSV 行
                            row_data = [
                                f"{current_sim_time:.3f}",
                                "Sentinel",
                                sentinel_res,
                                "N/A",
                                "N/A",
                                current_spe_str,
                                f"{lat:.2f}",  # Sentinel Time
                                "N/A",  # Expert Time
                                f"{current_cpu:.1f}"
                            ]

                            if is_anomaly:
                                self.stats['anomalies_detected'] += 1
                                print(f"[Sentinel @ {current_sim_time:.2f}s] Anomaly! SPE={spe:.4f}")

                                # 切换到专家模式
                                self.current_mode = "expert"
                                # 清空专家缓冲区
                                self.expert.buffer_vib = []
                                self.expert.buffer_audio = []
                                self.expert.buffer_speed = []
                                self.expert.buffer_current = []

                                # 专家模式需要重新积累数据，所以当前循环可能不再产生输出
                                # 写入当前这一行（Anomaly），然后跳出循环去积累专家数据
                                if output_path:
                                    with open(output_path, 'a', newline='', encoding='utf-8') as f:
                                        csv.writer(f).writerow(row_data)
                                break

                            # 如果正常，写入 CSV 并继续下一次微循环
                            if output_path:
                                with open(output_path, 'a', newline='', encoding='utf-8') as f:
                                    csv.writer(f).writerow(row_data)

                            results.append({
                                'time': current_sim_time,
                                'mode': 'sentinel',
                                'spe': spe,
                                'is_anomaly': is_anomaly,
                                'fault_class': None,
                                'confidence': None
                            })

                    else:  # Expert Mode
                        self.expert.add_data(vib_step, audio_step, speed_step, current_step)

                        while True:
                            process_monitor.cpu_percent(interval=None) # 1. 测量前重

                            class_id, class_name, confidence, lat = self.expert.process()

                            current_cpu = process_monitor.cpu_percent(interval=None) # 2. 测量后获取 (这里包含了特征提取的繁重计算，数值应比较大)

                            if class_id == -1:  # 数据不够
                                break

                            self.stats['expert_calls'] += 1
                            self.stats['faults_diagnosed'] += 1
                            print(f"[Expert @ {current_sim_time:.2f}s] Diagnosis: {class_name} ({confidence:.1%})")

                            # 准备 CSV 行
                            row_data = [
                                f"{current_sim_time:.3f}",
                                "Expert",
                                "Anomaly (Confirmed)",
                                class_name,
                                f"{confidence:.4f}",
                                triggering_spe,  # 显示触发这次诊断的SPE
                                "N/A",  # Sentinel Time
                                f"{lat:.2f}",  # Expert Time
                                f"{current_cpu:.1f}"
                            ]

                            if output_path:
                                with open(output_path, 'a', newline='', encoding='utf-8') as f:
                                    csv.writer(f).writerow(row_data)

                            results.append({
                                'time': current_sim_time,
                                'mode': 'expert',
                                'spe': None,
                                'is_anomaly': None,
                                'fault_class': class_name,
                                'confidence': confidence
                            })

                            # 诊断完成，切回哨兵
                            self.current_mode = "sentinel"
                            # 设置冷却时间 (模拟时间 + 3秒)
                            self.cooldown_until_sim_time = current_sim_time + self.cooldown_duration
                            print(f"[System] Cooldown until {self.cooldown_until_sim_time:.2f}s")
                            break  # 切回哨兵后，跳出循环等待新数据

                    # 更新总样本数
                    total_processed_samples += len(vib_step)

        # 处理结束
        duration = time.time() - start_time_wall
        print(f"\n>>> Streaming Finished in {duration:.2f}s")
        print(f"    Total Samples: {total_processed_samples}")

        if output_path:
            # 转换结果为DF并保存
            print(f"    Results saved to {output_path}")

        # 打印原有统计
        print(f"\n{'=' * 60}")
        print(f"    Sentinel calls: {self.stats['sentinel_calls']}")
        print(f"    Expert calls: {self.stats['expert_calls']}")
        print(f"    Anomalies: {self.stats['anomalies_detected']}")
        print(f"    Faults: {self.stats['faults_diagnosed']}")
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
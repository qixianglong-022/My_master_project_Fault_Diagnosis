#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : two_tier_diagnosis.py
@Desc    : 两级诊断策略系统 - 哨兵模式(异常检测) + 专家模式(故障分类)
@Author  : Auto-generated
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import onnxruntime as ort
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import deque
from datetime import datetime, timedelta

# 导入配置和工具
from config import Config, Ch4Config
from utils.feature_extractor import FeatureExtractor
from scipy.signal import decimate


class SentinelFeatureExtractor:
    """
    哨兵模式特征提取器（第三章特征工程）
    输入：原始传感器数据
    输出：特征序列 [Seq_Len, Feat_Dim] + 协变量 [2]
    """

    def __init__(self, config: Config):
        self.config = config
        self.extractor = FeatureExtractor(config)

        # 加载标准化参数
        self.scaler_path = config.SCALER_PATH
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                self.mean = scaler['mean']
                self.std = scaler['std']
        else:
            print(f"[Warn] Scaler not found at {self.scaler_path}, using default normalization")
            self.mean = None
            self.std = None

    def extract_features(self, vib_data: np.ndarray, audio_data: np.ndarray,
                         speed_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        提取哨兵模式特征

        Args:
            vib_data: [N, 4] 振动数据（4个通道）
            audio_data: [N] 声纹数据
            speed_data: [N] 转速数据

        Returns:
            x_feat: [Seq_Len, Feat_Dim] 特征序列
            cov: [2] 协变量 [v_mean, v2_mean]
        """
        feat_list = []

        # 1. 振动特征提取 (RMS + Kurtosis)
        for i in range(vib_data.shape[1]):
            f = self.extractor.extract_vib_features(vib_data[:, i])
            feat_list.append(f)

        # 2. 声纹特征提取 (MFCC + SF + ER)
        audio_feats = self.extractor.extract_audio_features(audio_data)
        feat_list.append(audio_feats)

        # 3. 拼接特征 -> [N_frames, Feat_Dim]
        x_feat = np.concatenate(feat_list, axis=1)

        # 4. 标准化
        if self.mean is not None and self.std is not None:
            x_feat = (x_feat - self.mean) / self.std

        # 5. 计算协变量（转速统计量）
        # 将转速从RPM转换为归一化值
        v_mean = np.mean(speed_data) / 3000.0  # 归一化
        v2_mean = np.mean(speed_data ** 2) / (3000.0 ** 2)
        cov = np.array([v_mean, v2_mean], dtype=np.float32)

        return x_feat, cov


class ExpertFeatureExtractor:
    """
    专家模式特征提取器（第四章特征工程）
    输入：原始传感器数据
    输出：多模态特征 (micro, macro, acoustic, current_spec, speed, load_proxy)
    """

    def __init__(self, config: Ch4Config):
        self.config = config
        self.extractor = FeatureExtractor(Config)

        # 降采样参数
        self.original_fs = Config.SAMPLE_RATE  # 51200
        self.decimation = 50
        self.target_fs = self.original_fs / self.decimation  # 1024 Hz
        self.fft_points = 1024
        self.hanning = np.hanning(self.fft_points)
        self.correction = 1.63

        # 窗口大小：1秒数据
        self.win_size_raw = self.fft_points * self.decimation  # 51200

        # 加载标准化参数
        self.scaler_path = os.path.join(config.CHECKPOINT_DIR, "scaler_ch4_soft.pkl")
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            print(f"[Warn] Ch4 scaler not found, using default normalization")
            self.scaler = None

    def _compute_fft(self, sig: np.ndarray) -> np.ndarray:
        """计算FFT频谱"""
        if len(sig) < self.fft_points:
            sig = np.pad(sig, (0, self.fft_points - len(sig)))

        # 移除直流
        sig = sig - np.mean(sig)
        sig = sig[:self.fft_points] * self.hanning

        spec = np.abs(np.fft.rfft(sig)) / (self.fft_points / 2)
        spec = spec * self.correction

        return spec[:self.fft_points // 2]

    def _robust_compute_frequency_hz(self, signal: np.ndarray, default_hz: float = 30.0) -> float:
        """鲁棒转速计算"""
        # 简单滤波
        signal = np.convolve(signal, np.ones(5) / 5, mode='same')

        _min, _max = np.min(signal), np.max(signal)
        if _max - _min < 0.5:
            return default_hz

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
            return default_hz

        intervals = np.diff(rising_indices)
        median_val = np.median(intervals)
        valid_intervals = intervals[np.abs(intervals - median_val) < 0.3 * median_val]

        if len(valid_intervals) == 0:
            avg_interval = median_val
        else:
            avg_interval = np.mean(valid_intervals)

        return self.original_fs / avg_interval

    def extract_features(self, vib_data: np.ndarray, audio_data: np.ndarray,
                         speed_data: np.ndarray, current_data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取专家模式特征

        Args:
            vib_data: [N] 振动数据（取第一个通道）
            audio_data: [N] 声纹数据
            speed_data: [N] 转速信号
            current_data: [N] 电流数据

        Returns:
            features: dict with keys: micro, macro, acoustic, current_spec, speed, load_proxy
        """
        # 需要至少1秒数据
        if len(vib_data) < self.win_size_raw:
            # 补零
            pad_len = self.win_size_raw - len(vib_data)
            vib_data = np.pad(vib_data, (0, pad_len), mode='constant')
            audio_data = np.pad(audio_data, (0, pad_len), mode='constant')
            speed_data = np.pad(speed_data, (0, pad_len), mode='constant')
            current_data = np.pad(current_data, (0, pad_len), mode='constant')

        # 截取1秒数据
        seg_vib = vib_data[:self.win_size_raw]
        seg_aud = audio_data[:self.win_size_raw]
        seg_spd = speed_data[:self.win_size_raw]
        seg_curr = current_data[:self.win_size_raw]

        # A. 计算物理转速 (Hz)
        hz = self._robust_compute_frequency_hz(seg_spd, default_hz=30.0)

        # B. Micro Stream (振动降采样 -> FFT)
        try:
            vib_low = decimate(seg_vib, self.decimation, zero_phase=True)
            fft_micro = self._compute_fft(vib_low)
        except Exception:
            fft_micro = np.zeros(self.config.FREQ_DIM, dtype=np.float32)

        # C. Macro Stream (原始高频振动 FFT)
        fft_macro = self._compute_fft(seg_vib[:self.fft_points])

        # D. Current Stream (电流降采样 -> FFT)
        try:
            curr_low = decimate(seg_curr, self.decimation, zero_phase=True)
            fft_curr_full = self._compute_fft(curr_low)
            fft_curr = fft_curr_full[:self.config.CURRENT_DIM]
        except Exception:
            fft_curr = np.zeros(self.config.CURRENT_DIM, dtype=np.float32)

        # E. Acoustic (MFCC) - 使用部分数据计算
        mfcc = np.mean(self.extractor.extract_audio_features(seg_aud[:16384]), axis=0)

        # F. Load Proxy (电流RMS归一化)
        curr_rms = np.sqrt(np.mean(seg_curr ** 2))
        if self.scaler and 'curr_rms_ref' in self.scaler:
            load_proxy = curr_rms / self.scaler['curr_rms_ref']
            load_proxy = np.clip(load_proxy, 0.0, 3.0)
        else:
            load_proxy = curr_rms / 100.0  # 简单归一化

        # 标准化
        if self.scaler:
            fft_micro = (fft_micro - self.scaler.get('micro_mean', 0)) / (self.scaler.get('micro_std', 1) + 1e-6)
            fft_macro = (fft_macro - self.scaler.get('macro_mean', 0)) / (self.scaler.get('macro_std', 1) + 1e-6)
            mfcc = (mfcc - self.scaler.get('ac_mean', 0)) / (self.scaler.get('ac_std', 1) + 1e-6)
            fft_curr = (fft_curr - self.scaler.get('curr_spec_mean', 0)) / (self.scaler.get('curr_spec_std', 1) + 1e-6)

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


class TwoTierDiagnosisSystem:
    """
    两级诊断系统
    - 哨兵模式：RDLinear-AD异常检测，10Hz刷新率
    - 专家模式：Phys-RDLinear故障分类，1Hz刷新率
    """

    def __init__(self,
                 sentinel_model_path: str,
                 expert_model_path: str,
                 sentinel_threshold_path: Optional[str] = None,
                 sentinel_config_path: Optional[str] = None):
        """
        Args:
            sentinel_model_path: 哨兵模式ONNX模型路径
            expert_model_path: 专家模式ONNX模型路径
            sentinel_threshold_path: 哨兵模式阈值文件路径
            sentinel_config_path: 哨兵模式配置路径（包含scaler等）
        """
        print(f"\n{'=' * 60}")
        print(f">>> Initializing Two-Tier Diagnosis System")
        print(f"{'=' * 60}\n")

        # 初始化哨兵模式
        print(">>> [Sentinel Mode] Loading model...")
        self.sentinel_session = ort.InferenceSession(sentinel_model_path, providers=['CPUExecutionProvider'])
        self.sentinel_input_names = [inp.name for inp in self.sentinel_session.get_inputs()]
        print(f"    Input names: {self.sentinel_input_names}")

        # 加载阈值
        if sentinel_threshold_path and os.path.exists(sentinel_threshold_path):
            self.sentinel_threshold = float(np.load(sentinel_threshold_path))
        else:
            # 尝试自动查找
            possible_paths = [
                Path(sentinel_model_path).parent / "threshold.npy",
                Path(Config.OUTPUT_DIR) / "threshold.npy"
            ]
            for path in possible_paths:
                if path.exists():
                    self.sentinel_threshold = float(np.load(path))
                    break
            else:
                print(f"    [Warn] Threshold not found, using default: 0.1")
                self.sentinel_threshold = 0.1

        print(f"    Threshold: {self.sentinel_threshold:.4f}")

        # 初始化专家模式
        print("\n>>> [Expert Mode] Loading model...")
        self.expert_session = ort.InferenceSession(expert_model_path, providers=['CPUExecutionProvider'])
        self.expert_input_names = [inp.name for inp in self.expert_session.get_inputs()]
        print(f"    Input names: {self.expert_input_names}")

        # 初始化特征提取器
        self.sentinel_fe = SentinelFeatureExtractor(Config)
        self.expert_fe = ExpertFeatureExtractor(Ch4Config())

        # 系统状态
        self.mode = "sentinel"  # "sentinel" or "expert"
        self.last_expert_time = None
        self.cooldown_period = 3.0  # 3秒冷却期
        self.expert_diagnosis_count = 0

        # 数据缓冲区
        self.sentinel_buffer = deque(maxlen=Config.WINDOW_SIZE * Config.HOP_LENGTH + Config.FRAME_SIZE)
        self.expert_buffer = deque(maxlen=51200)  # 1秒数据

        # 刷新率控制
        self.sentinel_interval = 0.1  # 10Hz = 0.1s
        self.expert_interval = 1.0  # 1Hz = 1.0s
        self.last_sentinel_time = 0
        self.last_expert_time_infer = 0

        # 故障类别映射
        self.fault_classes = ['HH', 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']

        print(f"\n>>> System Ready")
        print(f"    Sentinel Mode: {self.sentinel_interval * 1000:.0f}ms interval")
        print(f"    Expert Mode: {self.expert_interval * 1000:.0f}ms interval")
        print(f"    Cooldown Period: {self.cooldown_period}s")
        print(f"{'=' * 60}\n")

    def read_csv_data(self, csv_path: str) -> pd.DataFrame:
        """
        读取CSV数据文件
        格式：第一列时间，第二列振动1，后续为振动2_x、振动2_y、振动2_z、声音、转速、电流等
        """
        try:
            # 尝试读取，跳过可能的表头
            df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')

            # 如果第一行是表头，检查并处理
            if df.iloc[0, 0] == 'Time' or 'Time' in str(df.iloc[0, 0]):
                df = pd.read_csv(csv_path, skiprows=1, encoding='utf-8', errors='ignore')

            # 确保列数足够
            required_cols = 8  # 至少需要：时间、振动1、振动2_x、振动2_y、振动2_z、声音、转速、电流
            if df.shape[1] < required_cols:
                print(f"[Warn] CSV has only {df.shape[1]} columns, expected at least {required_cols}")

            return df
        except Exception as e:
            print(f"[Error] Failed to read CSV: {e}")
            raise

    def process_csv_stream(self, csv_path: str):
        """
        处理CSV数据流（模拟实时）
        """
        print(f"\n>>> Processing CSV stream: {os.path.basename(csv_path)}")

        df = self.read_csv_data(csv_path)

        # 解析列（假设格式：时间、振动1、振动2_x、振动2_y、振动2_z、声音、转速、电流）
        time_col = df.iloc[:, 0].values
        vib1 = df.iloc[:, 1].values if df.shape[1] > 1 else np.zeros(len(df))
        vib2_x = df.iloc[:, 2].values if df.shape[1] > 2 else np.zeros(len(df))
        vib2_y = df.iloc[:, 3].values if df.shape[1] > 3 else np.zeros(len(df))
        vib2_z = df.iloc[:, 4].values if df.shape[1] > 4 else np.zeros(len(df))
        audio = df.iloc[:, 5].values if df.shape[1] > 5 else np.zeros(len(df))
        speed = df.iloc[:, 6].values if df.shape[1] > 6 else np.zeros(len(df))
        current = df.iloc[:, 7].values if df.shape[1] > 7 else np.zeros(len(df))

        # 组合振动数据 [N, 4]
        vib_data = np.column_stack([vib1, vib2_x, vib2_y, vib2_z])

        # 模拟实时处理（按时间戳或固定步长）
        sample_rate = 51200  # 假设采样率
        chunk_size = int(sample_rate * 0.1)  # 100ms数据块

        total_samples = len(time_col)
        processed = 0

        print(f"    Total samples: {total_samples}")
        print(f"    Chunk size: {chunk_size}")
        print(f"\n>>> Starting diagnosis...\n")

        while processed < total_samples:
            end_idx = min(processed + chunk_size, total_samples)

            # 提取当前块
            chunk_vib = vib_data[processed:end_idx]
            chunk_audio = audio[processed:end_idx]
            chunk_speed = speed[processed:end_idx]
            chunk_current = current[processed:end_idx]

            # 更新缓冲区
            for i in range(len(chunk_vib)):
                self.sentinel_buffer.append({
                    'vib': chunk_vib[i],
                    'audio': chunk_audio[i],
                    'speed': chunk_speed[i]
                })
                self.expert_buffer.append({
                    'vib': chunk_vib[i, 0],  # 只用第一个振动通道
                    'audio': chunk_audio[i],
                    'speed': chunk_speed[i],
                    'current': chunk_current[i]
                })

            # 根据模式执行推理
            current_time = time.time()

            if self.mode == "sentinel":
                # 哨兵模式：10Hz刷新率
                if current_time - self.last_sentinel_time >= self.sentinel_interval:
                    if len(self.sentinel_buffer) >= Config.WINDOW_SIZE * Config.HOP_LENGTH + Config.FRAME_SIZE:
                        self._run_sentinel()
                    self.last_sentinel_time = current_time

            elif self.mode == "expert":
                # 专家模式：1Hz刷新率
                if current_time - self.last_expert_time_infer >= self.expert_interval:
                    if len(self.expert_buffer) >= 51200:
                        self._run_expert()
                    self.last_expert_time_infer = current_time

                # 检查是否返回哨兵模式
                if self.last_expert_time and (current_time - self.last_expert_time) >= self.cooldown_period:
                    self.mode = "sentinel"
                    print(f"[Mode Switch] Returning to Sentinel Mode (cooldown completed)")
                    self.last_expert_time = None

            processed = end_idx

            # 显示进度
            if processed % (chunk_size * 10) == 0:
                progress = 100 * processed / total_samples
                print(f"    Progress: {progress:.1f}% | Mode: {self.mode.upper()}")

        print(f"\n>>> Diagnosis completed")
        print(f"    Expert diagnoses: {self.expert_diagnosis_count}")

    def _run_sentinel(self):
        """运行哨兵模式推理"""
        # 准备数据
        buffer_array = np.array([item['vib'] for item in self.sentinel_buffer])
        audio_array = np.array([item['audio'] for item in self.sentinel_buffer])
        speed_array = np.array([item['speed'] for item in self.sentinel_buffer])

        # 特征提取
        x_feat, cov = self.sentinel_fe.extract_features(
            buffer_array, audio_array, speed_array
        )

        # 准备ONNX输入
        x_feat_batch = np.expand_dims(x_feat, axis=0).astype(np.float32)
        cov_batch = np.expand_dims(cov, axis=0).astype(np.float32)

        # 推理
        try:
            outputs = self.sentinel_session.run(None, {
                self.sentinel_input_names[0]: x_feat_batch,
                self.sentinel_input_names[1]: cov_batch
            })
            reconstruction = outputs[0]

            # 计算SPE
            spe = np.mean((x_feat_batch - reconstruction) ** 2)

            # 判断是否异常
            is_anomaly = spe > self.sentinel_threshold

            if is_anomaly:
                print(f"[Sentinel] SPE: {spe:.4f} > {self.sentinel_threshold:.4f} -> ANOMALY DETECTED!")
                # 切换到专家模式
                if self.mode == "sentinel":
                    self.mode = "expert"
                    self.last_expert_time = time.time()
                    self.last_expert_time_infer = time.time()
                    print(f"[Mode Switch] Switching to Expert Mode")
            else:
                # 静默模式，不打印正常状态
                pass

        except Exception as e:
            print(f"[Error] Sentinel inference failed: {e}")

    def _run_expert(self):
        """运行专家模式推理"""
        # 准备数据
        vib_array = np.array([item['vib'] for item in self.expert_buffer])
        audio_array = np.array([item['audio'] for item in self.expert_buffer])
        speed_array = np.array([item['speed'] for item in self.expert_buffer])
        current_array = np.array([item['current'] for item in self.expert_buffer])

        # 特征提取
        features = self.expert_fe.extract_features(
            vib_array, audio_array, speed_array, current_array
        )

        # 准备ONNX输入
        micro = np.expand_dims(features['micro'], axis=0).astype(np.float32)
        macro = np.expand_dims(features['macro'], axis=0).astype(np.float32)
        acoustic = np.expand_dims(features['acoustic'], axis=0).astype(np.float32)
        current_spec = np.expand_dims(features['current_spec'], axis=0).astype(np.float32)
        speed = np.expand_dims(features['speed'], axis=0).astype(np.float32)
        load_proxy = np.expand_dims(features['load_proxy'], axis=0).astype(np.float32)

        # 推理
        try:
            inputs = {
                self.expert_input_names[0]: micro,
                self.expert_input_names[1]: macro,
                self.expert_input_names[2]: acoustic,
                self.expert_input_names[3]: current_spec,
                self.expert_input_names[4]: speed,
                self.expert_input_names[5]: load_proxy
            }

            outputs = self.expert_session.run(None, inputs)
            logits = outputs[0]

            # 获取预测类别
            pred_class = np.argmax(logits[0])
            confidence = float(np.max(logits[0]))
            fault_name = self.fault_classes[pred_class]

            print(f"[Expert] Diagnosis: {fault_name} (Confidence: {confidence:.3f})")
            self.expert_diagnosis_count += 1

        except Exception as e:
            print(f"[Error] Expert inference failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Two-Tier Diagnosis System")
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV file path')
    parser.add_argument('--sentinel-model', type=str, required=True,
                        help='Sentinel mode ONNX model path')
    parser.add_argument('--expert-model', type=str, required=True,
                        help='Expert mode ONNX model path')
    parser.add_argument('--sentinel-threshold', type=str, default=None,
                        help='Sentinel threshold file path (optional)')
    parser.add_argument('--sentinel-config', type=str, default=None,
                        help='Sentinel config directory (optional)')

    args = parser.parse_args()

    # 初始化系统
    system = TwoTierDiagnosisSystem(
        sentinel_model_path=args.sentinel_model,
        expert_model_path=args.expert_
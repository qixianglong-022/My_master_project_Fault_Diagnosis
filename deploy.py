#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : deploy.py
@Desc    : 边缘端推理脚本，集成特征提取、Z-Score标准化与ONNX推理
"""
import os
import time
import json
import pickle
import numpy as np
import onnxruntime as ort
import pandas as pd
from utils.feature_extractor import FeatureExtractor
from edge_config import EdgeConfig  # [New] 引用独立配置


class EdgeSentinel:
    def __init__(self):
        print("[Init] Initializing Edge Sentinel...")

        # 1. 加载 ONNX 模型
        self.session = ort.InferenceSession(EdgeConfig.MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name_signal = self.session.get_inputs()[0].name
        self.input_name_cov = self.session.get_inputs()[1].name

        # 2. 加载 POT 阈值
        self.alarm_threshold = float(np.load(EdgeConfig.THRESHOLD_PATH))

        # 3. 加载自适应融合参数 (tau_base, th_vib)
        with open(EdgeConfig.PARAMS_PATH, 'r') as f:
            self.fusion_params = json.load(f)

        # 4. [新增] 加载 Z-Score 参数 (Mean, Std)
        with open(EdgeConfig.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        self.mean = scaler['mean']
        self.std = scaler['std']

        print(f"[Init] Params Loaded | Th: {self.alarm_threshold:.4f} | "
              f"Tau: {self.fusion_params['tau_base']:.4f}")

        # 5. 初始化特征提取器
        self.extractor = FeatureExtractor(EdgeConfig)

        # 6. 初始化缓冲区
        self.buffer_sensors = []
        self.buffer_speed = []

    def preprocess(self, sensor_window, speed_window):
        """
        处理单次推理的窗口数据
        """
        # --- 1. 特征提取 ---
        # 假设前4列是振动，最后1列是声纹
        raw_vib = sensor_window[:, :4]
        raw_audio = sensor_window[:, -1]

        feat_list = []

        # A. 振动 (RMS, Kurtosis)
        for i in range(4):
            f = self.extractor.extract_vib_features(raw_vib[:, i])
            feat_list.append(f)

        # B. 声纹 (MFCC)
        f_audio = self.extractor.extract_audio_features(raw_audio)
        feat_list.append(f_audio)

        # C. 拼接 -> [Seq_Len, 21]
        x_feat = np.concatenate(feat_list, axis=1)

        # --- 2. Z-Score 标准化 ---
        x_feat = (x_feat - self.mean) / self.std

        # 增加 Batch 维度 -> [1, Seq_Len, 21]
        x_feat = np.expand_dims(x_feat, axis=0).astype(np.float32)

        # --- 3. 协变量计算 ---
        v_bar = np.mean(speed_window)
        v2_bar = np.mean(speed_window ** 2)
        norm_scale = 3000.0
        cov = np.array([[v_bar / norm_scale, v2_bar / (norm_scale ** 2)]], dtype=np.float32)

        return x_feat, cov

    def infer(self, x, cov):
        start = time.time()
        res = self.session.run(None, {
            self.input_name_signal: x,
            self.input_name_cov: cov
        })[0]
        lat = (time.time() - start) * 1000

        # 计算 MSE (可扩展为自适应融合 SPE)
        diff = (x - res) ** 2
        spe = np.mean(diff)

        return spe, lat

    def process_stream(self, csv_path):
        """模拟实时数据流"""
        print(f"\n[Stream] Reading: {os.path.basename(csv_path)}")

        # 简单跳过 Header
        header_rows = 0
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if "Time (seconds)" in line:
                    header_rows = i + 1
                    break

        chunk_size = 512
        reader = pd.read_csv(
            csv_path, sep='\t', skiprows=header_rows, header=None,
            usecols=[EdgeConfig.IDX_SPEED] + EdgeConfig.IDX_SENSORS,
            chunksize=chunk_size, engine='c'
        )

        total_samples = 0
        anomalies = 0

        try:
            for chunk in reader:
                data = chunk.values
                # STM32 已经传上来了计算好的 RPM 值 (例如 1500.0)
                # ========================================================

                # 获取 RPM 列 (假设在第0列)
                # 建议加一个简单的异常值过滤，防止通信干扰产生 NaN 或 负数
                raw_rpms = data[:, 0]
                raw_rpms = np.where(raw_rpms < 0, 0, raw_rpms)  # 简单的物理约束
                raw_rpms = np.nan_to_num(raw_rpms, nan=0.0)

                speeds = raw_rpms
                sensors = data[:, 1:]  # 后面是传感器

                self.buffer_speed.extend(speeds)
                self.buffer_sensors.extend(sensors)

                # 缓冲区满，取出一个窗口
                # 注意：这里我们按照“帧”来攒数据。
                # 训练时的 Window_Size 是帧数(49)，对应的原始点数是 (49-1)*Hop + Frame ≈ 51200
                # 所以我们需要攒够这么多原始点，才能提取出足够的帧。

                # 计算需要的原始点数
                needed_points = (EdgeConfig.WINDOW_SIZE - 1) * EdgeConfig.HOP_LENGTH + EdgeConfig.FRAME_SIZE

                while len(self.buffer_sensors) >= needed_points:
                    win_s = np.array(self.buffer_sensors[:needed_points])
                    win_v = np.array(self.buffer_speed[:needed_points])

                    # 推理
                    input_x, input_cov = self.preprocess(win_s, win_v)
                    spe, latency = self.infer(input_x, input_cov)

                    # 判定
                    status = "NORMAL"
                    if spe > self.alarm_threshold:
                        status = "ANOMALY !!!"
                        anomalies += 1

                    print(f"Win: {total_samples // needed_points} | SPE: {spe:.4f} | Lat: {latency:.1f}ms | {status}")

                    # 滑动：移除一个 Stride 对应的原始点数
                    # Stride (帧) -> Stride (点)
                    stride_points = EdgeConfig.STRIDE * EdgeConfig.HOP_LENGTH

                    del self.buffer_sensors[:stride_points]
                    del self.buffer_speed[:stride_points]

                    total_samples += 1

        except KeyboardInterrupt:
            print("\n[Stop] Stopped by user.")


if __name__ == "__main__":
    # 请确保该文件存在，或修改路径
    test_file = "./test_data.txt"
    if os.path.exists(test_file):
        sentinel = EdgeSentinel()
        sentinel.process_stream(test_file)
    else:
        print(f"请指定有效的测试文件路径。")
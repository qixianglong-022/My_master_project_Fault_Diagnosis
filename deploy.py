#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : deploy.py//{DESC}/树莓派端推理脚本，加载 .onnx，实时输出异常得分
@Author  : Your Name
@Date    : $(date +%Y-%m-%d)
@Desc    : {DESC}
"""
import os
import time
import json
import numpy as np
import onnxruntime as ort
import pandas as pd
import pickle
from utils.feature_extractor import FeatureExtractor  # 调用手写的 Numpy 提取器
from config import Config

with open(Config.SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)
mean, std = scaler['mean'], scaler['std']

# 简单的 Config 代理，用于部署环境
class DeployConfig:
    SAMPLE_RATE = 51200
    FRAME_SIZE = 2048
    HOP_LENGTH = 1024
    N_MFCC = 13

    # 路径配置
    MODEL_PATH = "./checkpoints/rdlinear_edge.onnx"
    THRESHOLD_PATH = "./checkpoints/test_threshold.npy"
    PARAMS_PATH = "./checkpoints/fusion_params.json"

    # 窗口与通道
    WINDOW_SIZE = 50  # 对应训练时的 seq_len
    # 假设输入数据流: [Speed, Vib1, Vib2, Vib3, Vib4, Audio]
    # 需要根据实际情况切分


class EdgeSentinel:
    def __init__(self):
        print("[Init] Initializing Edge Sentinel...")

        # 1. 加载模型
        self.session = ort.InferenceSession(DeployConfig.MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name_signal = self.session.get_inputs()[0].name
        self.input_name_cov = self.session.get_inputs()[1].name

        # 2. 加载 POT 阈值 (总报警线)
        self.alarm_threshold = float(np.load(DeployConfig.THRESHOLD_PATH))

        # 3. 加载自适应融合参数 (tau_base, th_vib)
        with open(DeployConfig.PARAMS_PATH, 'r') as f:
            self.fusion_params = json.load(f)
        print(
            f"[Init] Loaded Params: tau_base={self.fusion_params['tau_base']:.4f}, th_vib={self.fusion_params['th_vib']:.4f}")

        # 4. 初始化高性能特征提取器
        self.extractor = FeatureExtractor(DeployConfig)

    def preprocess(self, sensor_window, speed_window):
        """
        处理单次推理的窗口数据
        :param sensor_window: [Raw_Len, 5] (假设前4列振动，最后1列声纹)
        :param speed_window: [Raw_Len] (原始转速点)
        """
        # ---------------------------------------------------------
        # 1. 特征提取 (Feature Extraction) - 使用 Numpy 加速版
        # ---------------------------------------------------------
        # 分离振动和声纹数据
        # 注意：这里的切片索引 [:, :4] 和 [:, -1] 需要与 Config.COL_INDICES 对应
        # 如果 Config 变了，这里也要变。建议后续写成通过 Config 获取索引。
        raw_vib = sensor_window[:, :4]
        raw_audio = sensor_window[:, -1]

        feat_list = []

        # A. 提取振动特征 (RMS, Kurtosis)
        # self.extractor 是在 __init__ 里初始化的 FeatureExtractor(DeployConfig)
        for i in range(4):  # 遍历4个振动通道
            # f shape: [Seq_Len, 2]
            f = self.extractor.extract_vib_features(raw_vib[:, i])
            feat_list.append(f)

        # B. 提取声纹特征 (MFCC)
        # f_audio shape: [Seq_Len, 13]
        f_audio = self.extractor.extract_audio_features(raw_audio)
        feat_list.append(f_audio)

        # C. 拼接所有特征 -> [Seq_Len, 21]
        x_feat = np.concatenate(feat_list, axis=1)

        # ---------------------------------------------------------
        # 2. Z-Score 标准化 (Normalization) - 核心修正
        # ---------------------------------------------------------
        # 必须使用训练集生成的均值和方差！
        # self.mean, self.std 是在 __init__ 里从 pickle 加载的
        x_feat = (x_feat - self.mean) / self.std

        # 增加 Batch 维度: [Seq_Len, 21] -> [1, Seq_Len, 21]
        x_feat = np.expand_dims(x_feat, axis=0).astype(np.float32)

        # ---------------------------------------------------------
        # 3. 协变量计算 (Covariates)
        # ---------------------------------------------------------
        # 计算窗口内的平均转速和平均动能
        v_bar = np.mean(speed_window)
        v2_bar = np.mean(speed_window ** 2)

        # 归一化 (保持与训练一致，训练如果除了3000，这里也要除)
        # 建议：如果训练也没做特殊归一化（只靠Linear层学），这里保持原值即可
        # 假设 config 中有归一化系数，或者直接传数值
        norm_scale = 3000.0
        cov = np.array([[v_bar / norm_scale, v2_bar / (norm_scale ** 2)]], dtype=np.float32)

        return x_feat, cov

    def infer(self, x, cov):
        # ONNX 推理
        start = time.time()
        res = self.session.run(None, {
            self.input_name_signal: x,
            self.input_name_cov: cov
        })[0]  # Recon
        lat = (time.time() - start) * 1000

        # 计算融合 SPE (这里要在 numpy 下复现 eval_metrics 的逻辑)
        # 为简洁，此处仅计算 MSE，实际部署应把 adaptive_fusion_spe 移植过来
        # 使用 self.fusion_params 进行加权
        diff = (x - res) ** 2
        spe = np.mean(diff)  # 简化的

        return spe, lat

    def process_stream(self, csv_path):
        """
        模拟实时数据流处理
        """
        print(f"\n[Stream] 开始读取数据流: {os.path.basename(csv_path)}")

        # 模拟：分块读取文件，假装是传感器发来的数据包
        # skiprows逻辑与 data_loader 类似，这里简化处理，假设已经去掉了header
        # 实际部署时，这里会替换为 sensor.read()

        # 自动跳过非数据行（简单的启发式：找第一列是数字的行）
        header_rows = 0
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if "Time (seconds)" in line:
                    header_rows = i + 1
                    break

        # 使用 pandas 分块读取 (chunksize)，模拟每秒钟发来的一批数据
        chunk_size = 512  # 每次读取512个点
        reader = pd.read_csv(
            csv_path,
            sep='\t',
            skiprows=header_rows,
            header=None,
            usecols=[EdgeConfig.IDX_SPEED] + EdgeConfig.IDX_SENSORS,
            chunksize=chunk_size,
            engine='c'
        )

        total_samples = 0
        anomalies = 0

        try:
            for chunk in reader:
                # data_chunk: [Chunk_Size, 1+N_Sensors]
                # 第一列是 Speed，后面是 Sensors
                data = chunk.values
                speeds = data[:, 0]
                sensors = data[:, 1:]

                # 将数据填入缓冲区
                self.buffer_speed.extend(speeds)
                self.buffer_sensors.extend(sensors)

                # 检查缓冲区是否填满一个窗口
                while len(self.buffer_sensors) >= EdgeConfig.WINDOW_SIZE:
                    # 取出一个窗口
                    win_s = self.buffer_sensors[:EdgeConfig.WINDOW_SIZE]
                    win_v = self.buffer_speed[:EdgeConfig.WINDOW_SIZE]

                    # === 核心处理流 ===
                    # 1. 预处理
                    input_x, input_cov = self.preprocess(win_s, win_v)

                    # 2. 推理
                    spe, latency = self.infer(input_x, input_cov)

                    # 3. 判定
                    status = "NORMAL"
                    if spe > self.threshold:
                        status = "ANOMALY !!!"
                        anomalies += 1

                    # 4. 打印日志 (模拟上位机显示)
                    print(f"Time: {total_samples / 51200:.2f}s | "
                          f"SPE: {spe:.6f} | Th: {self.threshold:.4f} | "
                          f"Lat: {latency:.2f}ms | {status}")

                    # 滑动窗口：移除 Stride 个旧数据
                    del self.buffer_sensors[:EdgeConfig.STRIDE]
                    del self.buffer_speed[:EdgeConfig.STRIDE]

                    total_samples += EdgeConfig.STRIDE

        except KeyboardInterrupt:
            print("\n[Stop] 用户停止监测")

        print(f"\n[Summary] 监测结束。共检测窗口: {total_samples // EdgeConfig.STRIDE}, 发现异常: {anomalies}")


if __name__ == "__main__":
    # 简单的文件存在性检查
    if not os.path.exists(EdgeConfig.TEST_FILE):
        print(f"请修改 EdgeConfig.TEST_FILE 为你电脑上真实存在的 txt 文件路径！")
    else:
        sentinel = EdgeSentinel()
        sentinel.process_stream(EdgeConfig.TEST_FILE)
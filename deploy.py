#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : deploy.py//{DESC}/树莓派端推理脚本，加载 .onnx，实时输出异常得分
@Author  : Your Name
@Date    : $(date +%Y-%m-%d)
@Desc    : {DESC}
"""
# deploy.py
import os
import time
import numpy as np
import onnxruntime as ort
import pandas as pd


# 引入配置，但为了部署解耦，我们也可以在这里重新定义关键参数
# 这样 deploy.py 就可以作为一个独立文件拷贝到树莓派，不需要带上整个工程
class EdgeConfig:
    # === 1. 模型与阈值路径 ===
    # 这些文件是 train.py 运行后生成的
    MODEL_PATH = "./checkpoints/rdlinear_edge.onnx"
    THRESHOLD_PATH = "./checkpoints/test_threshold.npy"

    # === 2. 物理参数 (必须与训练时 config.py 完全一致) ===
    # 请务必检查 train.py 训练时实际使用的窗口大小！
    WINDOW_SIZE = 1024  # 示例：1024点 (约20ms数据)
    STRIDE = 1024  # 部署时通常无重叠实时处理，或者 50% 重叠

    # 通道索引 (对应原始 txt/csv 的列号)
    # 假设: 0=Time, 1=Speed, 8=Vib, 18=Audio (请根据实际情况修改)
    IDX_SPEED = 1
    IDX_SENSORS = [8, 18]  # [电机振动, 噪声]

    # === 3. 模拟数据源 ===
    # 指定一个测试文件来模拟实时流
    TEST_FILE = r"C:\毕业材料_齐祥龙\电机故障数据集\实验台数据采集\第5组——电机健康状态：电压不平衡（VU）\VU-0-3.txt"


class EdgeSentinel:
    """
    边缘哨兵：运行在树莓派上的核心检测类
    """

    def __init__(self):
        print("[Init] 正在初始化边缘哨兵系统...")

        # 1. 加载 ONNX 模型
        if not os.path.exists(EdgeConfig.MODEL_PATH):
            raise FileNotFoundError(f"找不到模型文件: {EdgeConfig.MODEL_PATH}")

        # 使用 CPU 提供程序 (树莓派通常用 CPU)
        self.session = ort.InferenceSession(EdgeConfig.MODEL_PATH, providers=['CPUExecutionProvider'])
        print("[Init] ONNX 模型加载成功")

        # 获取输入输出名称
        self.input_name_signal = self.session.get_inputs()[0].name
        self.input_name_cov = self.session.get_inputs()[1].name
        self.output_name = self.session.get_outputs()[0].name

        # 2. 加载报警阈值
        if os.path.exists(EdgeConfig.THRESHOLD_PATH):
            self.threshold = float(np.load(EdgeConfig.THRESHOLD_PATH))
            print(f"[Init] 加载动态阈值: {self.threshold:.6f}")
        else:
            print("[Warning] 未找到阈值文件，使用默认值 1.0 (仅供调试)")
            self.threshold = 1.0

        # 3. 初始化滑动窗口缓冲区
        # 我们需要缓存 WINDOW_SIZE 个历史点
        self.buffer_sensors = []
        self.buffer_speed = []

        print("[Init] 系统就绪，等待数据流...")

    def preprocess(self, sensor_window, speed_window):
        """
        物理特征工程：与 data_loader.py 保持严格一致
        """
        # 1. 转换为 numpy float32
        x = np.array(sensor_window, dtype=np.float32)
        s = np.array(speed_window, dtype=np.float32)

        # 2. 计算转速协变量 [Mean, Mean_Square]
        v_bar = np.mean(s)
        v2_bar = np.mean(s ** 2)
        cov = np.array([v_bar, v2_bar], dtype=np.float32)

        # 3. 增加 Batch 维度 [Length, Chan] -> [1, Length, Chan]
        x = np.expand_dims(x, axis=0)
        cov = np.expand_dims(cov, axis=0)

        return x, cov

    def infer(self, x, cov):
        """
        执行推理并计算异常得分
        """
        # 1. ONNX 推理
        start_time = time.time()

        outputs = self.session.run(
            [self.output_name],
            {
                self.input_name_signal: x,
                self.input_name_cov: cov
            }
        )
        recon = outputs[0]  # [1, Length, Chan]

        inference_time = (time.time() - start_time) * 1000  # ms

        # 2. 计算 SPE (Squared Prediction Error)
        # 简单的 MSE: mean((input - recon)^2)
        diff = x - recon
        spe = np.mean(np.square(diff))

        return spe, inference_time

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
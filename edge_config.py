# edge_config.py
# 这个文件是专门给树莓派用的，包含了运行所需的全部参数和路径。
class EdgeConfig:
    # ================= 1. 物理参数 =================
    # 必须与训练时的 Config 严格一致
    SAMPLE_RATE = 51200
    FRAME_SIZE = 2048  # FFT 窗口大小
    HOP_LENGTH = 1024  # 帧移
    N_MFCC = 13  # MFCC 系数数量

    # 窗口长度 (单位: 帧)
    # 对应训练时的 seq_len = (51200 - 2048) // 1024 + 1 ≈ 49
    # 请务必检查训练日志中的 Input Shape 确认此值
    WINDOW_SIZE = 49

    # 滑动步长 (部署时通常不重叠，或者 50% 重叠)
    # 这里设置为 WINDOW_SIZE 表示无重叠处理
    STRIDE = 49

    # ================= 2. 路径配置 =================
    # 这些文件由 train.py 生成，需拷贝到树莓派
    MODEL_PATH = "./checkpoints/rdlinear_edge.onnx"
    THRESHOLD_PATH = "./checkpoints/test_threshold.npy"
    PARAMS_PATH = "./checkpoints/fusion_params.json"
    SCALER_PATH = "./checkpoints/scaler_params.pkl"  # [新增] Z-Score 参数

    # ================= 3. 传感器通道定义 =================
    # 对应 CSV 文件的列索引
    IDX_SPEED = 1
    # 假设前4个是振动，后1个是声纹 (需与训练 data_loader 的拼接顺序一致)
    IDX_SENSORS = [8, 10, 11, 12, 20]
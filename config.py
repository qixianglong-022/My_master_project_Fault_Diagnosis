# config.py
import os


class Config:
    # ================= 1. 路径设置 =================
    # 原子数据目录
    ATOMIC_DATA_DIR = "./processed_data_atomic"

    # 结果输出根目录
    OUTPUT_DIR = "./checkpoints"

    SCALER_PATH = "./checkpoints/scaler_params.pkl"

    # ================= 2. 数据集定义 =================
    # 映射简写代码到实际文件夹名
    DATA_DOMAINS = {
        'HH': '第1组——电机健康状态：健康（HH）',
        'RU': '第2组——电机健康状态：转子不平衡（RU）',
        'RM': '第3组——电机健康状态：转子不对中故障（RM）',
        'SW': '第4组——电机健康状态：定子绕组故障（SW）',
        'VU': '第5组——电机健康状态：电压不平衡（VU）',
        'BR': '第6组——电机健康状态：轴弯曲（转子弯曲，BR）',
        'KA': '第7组——电机健康状态：转子条断裂故障（KA）',
        'FB': '第8组——电机健康状态：轴承故障（FB）',
    }

    # ================= 3. 训练集定义 (Source Domain) =================
    # 200kg (2) 下的 Speed 1, 3, 5, 7
    # 格式: (Load, Speed)
    TRAIN_ATOMS = [
        ('2', '1'), ('2', '3'), ('2', '5'), ('2', '7')
    ]

    # ================= 4. 测试集定义 (Dynamic Target Domain) =================
    # 这里只定义“当前”要跑的测试集。
    # 你可以手动修改这里来单跑某个工况，或者由外部脚本动态覆盖它。

    # 默认值：手动 Debug 时用，例如只想看 200kg 下 30Hz 的表现
    TEST_TASK_NAME = "Manual_Debug_200kg_30Hz"
    TEST_ATOMS = [('2', '2')]

    # ================= 5. 物理通道定义 =================
    # 根据TXT的 Legend 确定列索引
    # 原始列：Time(0), 转速(1), ..., 电机振动(8), ...,测试轴承X(10), 测试轴承Y(11), 测试轴承Z(12)..., 噪声1-1(20)...
    # 注意：pandas读取时如果包含Time列，索引如下：

    # 核心协变量通道
    COL_INDEX_SPEED = 1  # 转速

    # 核心监测通道 (模型输入 x)
    # [振动通道]: 8(电机振动), 10(轴承X), 11(轴承Y), 12(轴承Z)
    COL_INDICES_VIB = [8, 10, 11, 12]

    # [声纹通道]: 20(噪声1-1 - 假设这是信噪比最好的)
    COL_INDICES_AUDIO = [20]

    # 自动合并所有输入通道 (用于 data_loader 读取)
    COL_INDICES_X = COL_INDICES_VIB + COL_INDICES_AUDIO

    # ================= 6. 信号处理参数 =================
    SAMPLE_RATE = 51200

    # 原始数据窗口 (1秒)
    RAW_WINDOW_SIZE = 51200

    # 特征提取的“帧”参数
    # Frame Size: 计算一次特征的时间窗，例如 2048点 (40ms)
    FRAME_SIZE = 2048
    # Hop Length: 帧移，例如 1024点 (20ms, 50%重叠)
    HOP_LENGTH = 1024

    # 训练时的步长 (单位: 帧)
    # 比如设为 1，表示每隔一帧取一个窗口 (最大化数据利用率)
    # 或者设为 WINDOW_SIZE (无重叠)
    TRAIN_STRIDE_FRAMES = 1

    # 验证/测试时的步长
    #TEST_STRIDE_FRAMES = RAW_WINDOW_SIZE  # 通常测试时不重叠

    # 最终输入模型的序列长度 (Seq_Len)
    # 51200 / 1024 ≈ 50 帧
    WINDOW_SIZE = (RAW_WINDOW_SIZE - FRAME_SIZE) // HOP_LENGTH + 1

    # 无重叠切片原始数据 (训练时)
    STRIDE = RAW_WINDOW_SIZE

    # 特征维度定义
    # 振动: 2个特征 (RMS, Kurtosis) * 通道数 = 8
    FEAT_DIM_VIB = 2 * len(COL_INDICES_VIB)
    # 声纹: 13个 MFCC 系数 * 通道数 = 13
    N_MFCC = 13
    FEAT_DIM_AUDIO = N_MFCC * len(COL_INDICES_AUDIO)

    # 模型总输入维度 (enc_in) = 13 + 8 = 21
    ENC_IN = FEAT_DIM_VIB + FEAT_DIM_AUDIO

    # ================= 7. 训练超参数 =================
    # [修正] 设置为 None 以加载该工况下的所有文件 (含变速工况)
    # 如果内存不足 (OOM)，可改为整数 (如 4) 并手动挑选代表性文件
    LIMIT_FILES = None

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    EPOCHS = 100

    # ================= 8. 实验控制参数 (New) =================
    # 模型选择: 'RDLinear', 'DLinear', 'LSTM_AE','Informer', 'Autoformer', 'midruleDLinear', 'TiDE'
    # MODEL_NAME = 'RDLinear' # ours
    # MODEL_NAME = 'LSTM_AE'
    MODEL_NAME = 'RDLinear'

    # 消融实验开关 (Ablation Flags)
    USE_REVIN = True  # 是否使用 RevIN
    USE_SPEED = True  # 是否使用转速引导 Trend

    # 噪声测试 (只在测试时生效)
    TEST_NOISE_SNR = None  # None表示无噪声，数字表示信噪比(dB)

    # 自适应融合参数保存路径
    # 存储 tau_base, th_vib 等统计值
    FUSION_PARAMS_PATH = "./checkpoints/fusion_params.json"
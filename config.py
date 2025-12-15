import os

class Config:
    # ================= 1. 基础路径 =================
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_ROOT = r"C:\毕业材料_齐祥龙\电机故障数据集\实验台数据采集"
    ATOMIC_DATA_DIR = os.path.join(PROJECT_ROOT, "processed_data_atomic")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_params.pkl") # 归一化参数保存路径
    FUSION_PARAMS_PATH = os.path.join(OUTPUT_DIR, "fusion_params.json") # 自适应融合参数保存路径

    # ================= 2. 映射关系 (物理量 -> 文件ID) =================
    LOAD_MAP = {0: '0', 200: '2', 400: '4'}
    SPEED_MAP = {
        '15': '1', '30': '2', '45': '3', '60': '4',
        '15-45': '5', '30-60': '6', '45-15': '7', '60-30': '8'
    }

    # ================= 3. 数据集定义 (物理工况) =================
    # 训练集: 200kg 下的 4 种工况
    TRAIN_ATOMS = [
        (200, '15'), (200, '45'), (200, '15-45'), (200, '45-15')
    ]

    # 测试集列表: 程序会自动循环跑这些工况，并生成对应的文件夹
    # 示例: 测试 400kg 下的所有转速工况
    TEST_ATOMS = [
        (400, '15'), (400, '45'), (400, '15-45'), (400, '45-15')
    ]

    # 测试目标故障类型
    # 可以在这里显式指定要测试哪些故障，而不是默认 FB
    # 可选值对应 DATA_DOMAINS 的 Key: 'RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB'
    # TEST_FAULT_TYPES = ['FB']
    # 全量测试
    TEST_FAULT_TYPES = ['RU', 'RM', 'SW', 'VU', 'BR', 'KA', 'FB']
    # 如果想同时测不平衡和轴承: TEST_FAULT_TYPES = ['RU', 'FB']

    # ================= 4. 物理通道与维度 =================
    # 原始列：Time(0), 转速(1), ..., 电机振动(8), ...,测试轴承X(10), 测试轴承Y(11), 测试轴承Z(12)..., 噪声1-1(20)...
    # 核心协变量通道
    COL_INDEX_SPEED = 1  # 转速
    # [振动通道]: 8(电机振动), 10(轴承X), 11(轴承Y), 12(轴承Z)
    COL_INDICES_VIB = [8, 10, 11, 12]
    # [声纹通道]: 20(噪声1-1 - 假设这是信噪比最好的)
    COL_INDICES_AUDIO = [20]
    # 自动合并所有输入通道 (用于 data_loader 读取)
    COL_INDICES_X = COL_INDICES_VIB + COL_INDICES_AUDIO

    # ================= 5. 特征维度定义 =================
    USE_MFCC = True
    USE_LFCC = False

    # 声纹特征结构: [MFCC(13), Spectral_Flatness(1), Energy_Ratio(1)]
    N_MFCC = 13
    N_EXTRA_AUDIO = 2  # SF + EnergyRatio

    # 振动特征结构: [RMS(1), Kurtosis(1)]
    N_VIB_FEAT = 2

    # 计算总维度
    FEAT_DIM_AUDIO = (N_MFCC + N_EXTRA_AUDIO) * len([20])  # 假设1个声纹通道
    FEAT_DIM_VIB = N_VIB_FEAT * len([8, 10, 11, 12])  # 4个振动通道

    # 输入总维度
    ENC_IN = FEAT_DIM_VIB + FEAT_DIM_AUDIO

    # [索引辅助] 用于 anomaly.py 快速定位
    # 振动特征在 Tensor 中的范围: [0, FEAT_DIM_VIB)
    # 声纹特征在 Tensor 中的范围: [FEAT_DIM_VIB, END)
    # 其中声纹的 SF 特征位于声纹向量的第 13 位 (索引从0开始是13)
    IDX_AUDIO_SF = 13
    IDX_AUDIO_ER = 14

    # ================= 6. 数据处理参数 =================
    SAMPLE_RATE = 51200
    RAW_WINDOW_SIZE = 51200
    FRAME_SIZE = 8192
    HOP_LENGTH = 1024
    WINDOW_SIZE = (RAW_WINDOW_SIZE - FRAME_SIZE) // HOP_LENGTH + 1
    STRIDE = RAW_WINDOW_SIZE
    PREDICT_STEP = 3

    # ================= 7. 模型与训练 =================
    # 模型选择: 'RDLinear', 'DLinear', 'LSTM_AE','Informer', 'Autoformer', 'midruleDLinear', 'TiDE'
    MODEL_NAME = 'LSTM_AE'
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    PRED_LEN = 0 # 预测步长，如果PRED_LEN = 0 或 None，则默认为重构任务 (Output Length = Window Size)

    # 消融实验
    USE_REVIN = False
    USE_SPEED = True
    # 噪声测试 (None 或 dB值)
    TEST_NOISE_SNR = None

    # 路径映射 (保持之前的 DATA_DOMAINS)
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

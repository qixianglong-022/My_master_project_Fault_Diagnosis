import os
import argparse
import sys
import json
import torch
import numpy as np
from copy import deepcopy

# 导入默认配置
from config import Config
from utils.tools import set_seed

# 导入业务逻辑
from train import train_model
from run_evaluation import run_evaluation_loop
from preprocess_atomic import process_all

# ==============================================================================
# 1. 实验场景注册表 (SCENARIO REGISTRY) - 论文实验设计专用
# ==============================================================================

# 定义通用的训练集：200kg 下的 4 种典型工况 (覆盖低速、高速、加速、减速)
# Source Domain: Load 200kg
SOURCE_TRAIN_ATOMS = [
    (200, '15'), (200, '45'), (200, '15-45'), (200, '45-15')
]

# 定义所有 8 种工况的列表 (用于全工况测试)
ALL_SPEED_ATOMS_CODES = [
    '15', '30', '45', '60',       # 稳态
    '15-45', '30-60', '45-15', '60-30' # 变态
]

SCENARIOS = {
    # -------------------------------------------------------
    # A. 基础实验 (Baseline)
    # 目的：验证模型在同分布下的故障检测能力，用于调参
    # -------------------------------------------------------
    'baseline': {
        'description': '基础实验: 200kg同工况训练与测试 (Sanity Check)',
        'train_atoms': SOURCE_TRAIN_ATOMS,
        # 测试集与训练集工况一致 (但包含故障数据)
        'test_atoms':  SOURCE_TRAIN_ATOMS
    },

    # -------------------------------------------------------
    # B. 泛化性实验 - 阶段一: 同负载，未见转速 (Intra-Load Transfer)
    # 目的：验证对未见转速(30Hz插值, 60Hz外推)的适应能力
    # -------------------------------------------------------
    'transfer_speed': {
        'description': '转速迁移: 200kg下测试未见过的 4 种工况 (30, 60, 30-60, 60-30)',
        'train_atoms': SOURCE_TRAIN_ATOMS,
        'test_atoms': [
            (200, '30'), (200, '60'), (200, '30-60'), (200, '60-30')
        ]
    },

    # -------------------------------------------------------
    # C. 泛化性实验 - 阶段二: 跨负载 (Inter-Load Transfer)
    # 目的：验证模型在负载发生显著漂移(0kg, 400kg)时的鲁棒性
    # -------------------------------------------------------
    'transfer_load_0kg': {
        'description': '轻载迁移: 训练于200kg，测试于 0kg (全8种工况)',
        'train_atoms': SOURCE_TRAIN_ATOMS,
        # 生成 (0, '15'), (0, '30')... 等 8 个组合
        'test_atoms':  [(0, speed) for speed in ALL_SPEED_ATOMS_CODES]
    },

    'transfer_load_400kg': {
        'description': '重载迁移: 训练于200kg，测试于 400kg (全8种工况)',
        'train_atoms': SOURCE_TRAIN_ATOMS,
        # 生成 (400, '15'), (400, '30')... 等 8 个组合
        'test_atoms':  [(400, speed) for speed in ALL_SPEED_ATOMS_CODES]
    }
}

def apply_overrides(args):
    """
    将命令行参数和场景配置应用到 Config 全局单例上
    """
    # 1. 应用场景配置 (Dataset Atoms)
    if args.scenario in SCENARIOS:
        scene = SCENARIOS[args.scenario]
        print(f">>> Loading Scenario: [{args.scenario}]")
        print(f"    Desc: {scene['description']}")
        # 覆盖 Config 中的原子工况定义
        if 'train_atoms' in scene: Config.TRAIN_ATOMS = scene['train_atoms']
        if 'test_atoms' in scene:  Config.TEST_ATOMS = scene['test_atoms']
    else:
        print(f"[Warn] Scenario '{args.scenario}' not found, using Config default.")

    # 2. [新增] 应用模型配置
    if args.model_name:
        Config.MODEL_NAME = args.model_name

    # 3. 应用消融实验配置 (Ablation)
    if args.ablation == 'full':
        Config.USE_SPEED = True
        Config.USE_REVIN = True
    elif args.ablation == 'no_speed':
        Config.USE_SPEED = False  # 禁用转速引导
        Config.USE_REVIN = True
    elif args.ablation == 'no_revin':
        Config.USE_SPEED = True
        Config.USE_REVIN = False  # 禁用 RevIN

    # 4. 应用噪声配置
    # 注意：args.noise_snr 为 None 时表示不加噪
    Config.TEST_NOISE_SNR = args.noise_snr

    # 5. 常规超参覆盖
    if args.lr: Config.LEARNING_RATE = args.lr
    if args.batch_size: Config.BATCH_SIZE = args.batch_size
    if args.epochs: Config.EPOCHS = args.epochs

    # 6. 实验命名自动化 (包含模型名)
    if args.exp_name is None:
        # 自动生成名称结构: Scenario_ModelName_Ablation_Noise
        # 例如: baseline_RDLinear_full
        noise_str = f"_noise{args.noise_snr}dB" if args.noise_snr is not None else ""
        args.exp_name = f"{args.scenario}_{Config.MODEL_NAME}_{args.ablation}{noise_str}"

    # 更新路径
    Config.OUTPUT_DIR = os.path.join(Config.PROJECT_ROOT, "checkpoints", args.exp_name)
    Config.SCALER_PATH = os.path.join(Config.OUTPUT_DIR, "scaler_params.pkl")
    Config.FUSION_PARAMS_PATH = os.path.join(Config.OUTPUT_DIR, "fusion_params.json")


def dump_experiment_meta(args):
    """
    保留证据：将当前运行的所有物理定义和参数保存到日志
    解决了你担心的"物理映射关系没有保留"的问题
    """
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    meta_path = os.path.join(Config.OUTPUT_DIR, "experiment_meta.txt")

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write("=== Experiment Configuration ===\n")
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Scenario: {args.scenario}\n")
        f.write(f"Ablation: {args.ablation}\n")
        f.write(f"Noise SNR: {Config.TEST_NOISE_SNR}\n")
        f.write("\n=== Physical Definitions (From Config) ===\n")
        f.write(f"Vibration Channels: {Config.COL_INDICES_VIB}\n")
        f.write(f"Audio Channels: {Config.COL_INDICES_AUDIO}\n")
        f.write(f"Speed Channel: {Config.COL_INDEX_SPEED}\n")
        f.write(f"Feature Dim (Enc_In): {Config.ENC_IN}\n")
        f.write("\n=== Dataset Atoms ===\n")
        f.write(f"Train Atoms: {Config.TRAIN_ATOMS}\n")
        f.write(f"Test Atoms:  {Config.TEST_ATOMS}\n")
        f.write("\n=== Hyper Parameters ===\n")
        f.write(f"LR: {Config.LEARNING_RATE}\n")
        f.write(f"Batch: {Config.BATCH_SIZE}\n")
        f.write(f"Use RevIN: {Config.USE_REVIN}\n")
        f.write(f"Use Speed: {Config.USE_SPEED}\n")

    print(f"[Info] 实验配置快照已保存: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="RDLinear-AD 自动化实验台")

    # --- 核心控制 ---
    parser.add_argument('--mode', type=str, default='all', choices=['preprocess', 'train', 'eval', 'all'])
    parser.add_argument('--exp_name', type=str, default=None, help='实验文件夹名(留空则自动生成)')
    parser.add_argument('--model_name', type=str, default=None,
                        choices=['RDLinear', 'DLinear', 'LSTMAE', 'TiDE', 'Transformer'],
                        help='选择模型架构')

    # --- 场景与消融 (The Elegant Part) ---
    parser.add_argument('--scenario', type=str, default='baseline', choices=SCENARIOS.keys(),
                        help='选择实验场景(数据集组合)')
    parser.add_argument('--ablation', type=str, default='full', choices=['full', 'no_speed', 'no_revin'],
                        help='消融实验模式')
    parser.add_argument('--noise_snr', type=float, default=None,
                        help='测试集注入噪声的SNR(dB)，不填则不加噪')

    # --- 基础超参 ---
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', type=str, default='0')

    args = parser.parse_args()

    # 1. 环境设定
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_seed(args.seed)

    # 2. 应用配置覆盖 (核心逻辑)
    apply_overrides(args)

    # 3. 保存快照 (解决你的后顾之忧)
    if args.mode in ['train', 'all']:
        dump_experiment_meta(args)

    print(f"==================================================")
    print(f"   Running: {Config.OUTPUT_DIR}")
    print(f"   Ablation: Speed={Config.USE_SPEED}, RevIN={Config.USE_REVIN}")
    print(f"   Noise: {Config.TEST_NOISE_SNR} dB")
    print(f"==================================================")

    # --- 阶段一：数据预处理 ---
    # 如果指定了 preprocess 模式，或者 all 模式且数据目录为空，则运行
    need_preprocess = (args.mode == 'preprocess')
    if args.mode == 'all':
        # 智能检测：如果目标文件夹是空的，自动触发预处理
        if not os.path.exists(Config.ATOMIC_DATA_DIR) or not os.listdir(Config.ATOMIC_DATA_DIR):
            print("[Auto] 检测到预处理数据缺失，即将开始预处理...")
            need_preprocess = True

    if need_preprocess:
        print("\n>>> [Step 1] Starting Data Preprocessing...")
        process_all()  # 调用 preprocess_atomic.py 中的主逻辑
        if args.mode == 'preprocess':
            print("预处理完成。退出。")
            return

    # --- 阶段二：模型训练 ---
    if args.mode in ['train', 'all']:
        print("\n>>> [Step 2] Starting Model Training...")
        # 为了记录方便，把当前的 Config 配置保存到目录下
        with open(os.path.join(Config.OUTPUT_DIR, "run_config.txt"), "w") as f:
            f.write(str(vars(args)))

        train_model()

    # --- 阶段三：模型评估 ---
    if args.mode in ['eval', 'all']:
        print("\n>>> [Step 3] Starting Evaluation...")
        # 评估前检查是否有模型文件
        ckpt_path = os.path.join(Config.OUTPUT_DIR, 'checkpoint.pth')
        if not os.path.exists(ckpt_path):
            print(f"[Error] 未找到模型文件: {ckpt_path}，无法进行评估。")
            sys.exit(1)

        run_evaluation_loop()

    print(f"\n>>> All Tasks for [{args.exp_name}] Completed Successfully!")


if __name__ == '__main__':
    main()
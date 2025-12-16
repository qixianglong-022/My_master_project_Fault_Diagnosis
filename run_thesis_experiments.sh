#!/bin/bash
export PYTHONHASHSEED=42
# ==============================================================================
# 硕士毕业论文最终实验自动化脚本 (Thesis Final Experiments)
# 核心策略: RDLinear + No RevIN + 物理阻断 (Physics Constraint)
# ==============================================================================

# 1. 定义实验名称 (与代码中的修改对应)
EXP_NAME="Thesis_Final_Physics_Constraint_AllFault_Transformer"

# 2. 核心参数配置
# --ablation no_revin: 关键！禁用 RevIN，防止能量特征泄露
# --model_name RDLinear: 使用修改过 Trend 分支的物理引导模型
# COMMON_ARGS="--model_name RDLinear --ablation no_revin --batch_size 64"
# COMMON_ARGS="--model_name DLinear --batch_size 64"
# COMMON_ARGS="--model_name LSTMAE --batch_size 64"
# COMMON_ARGS="--model_name TiDE --batch_size 64"
COMMON_ARGS="--model_name Transformer --batch_size 64"

# Python 解释器 (防止多环境冲突)
PYTHON="python"

echo "========================================================"
echo "   硕士毕业论文全流程实验自动化脚本"
echo "   Strategy: Physics-Constrained RDLinear (No-RevIN)"
echo "   Experiment Name: ${EXP_NAME}"
echo "========================================================"

# --------------------------------------------------------
# Step 1: 训练源域模型 (Training Phase)
# Domain: 200kg (覆盖 15Hz, 45Hz, 加减速)
# --------------------------------------------------------
echo ""
echo ">>> [Step 1] Training Source Model (200kg)..."
echo "    Log Dir: checkpoints/${EXP_NAME}"

$PYTHON main.py --mode train \
    --scenario baseline \
    --exp_name ${EXP_NAME} \
    $COMMON_ARGS

# 检查训练是否成功
if [ $? -ne 0 ]; then
    echo "[Error] Training failed! Please check the logs."
    exit 1
fi

# --------------------------------------------------------
# Step 2: 评估 - 基础实验 (Evaluation - Baseline)
# Domain: 200kg (Seen Conditions - 验证拟合能力)
# --------------------------------------------------------
echo ""
echo ">>> [Step 2.1] Evaluating Baseline (200kg Seen)..."
$PYTHON main.py --mode eval \
    --scenario baseline \
    --exp_name ${EXP_NAME} \
    $COMMON_ARGS

# --------------------------------------------------------
# Step 3: 评估 - 转速迁移 (Evaluation - Transfer Speed)
# Domain: 200kg (Unseen Speeds: 30Hz, 60Hz, 30-60Hz...)
# --------------------------------------------------------
echo ""
echo ">>> [Step 2.2] Evaluating Speed Transfer (200kg Unseen)..."
$PYTHON main.py --mode eval \
    --scenario transfer_speed \
    --exp_name ${EXP_NAME} \
    $COMMON_ARGS

# --------------------------------------------------------
# Step 4: 评估 - 轻载迁移 (Evaluation - Load Transfer 0kg)
# Domain: 0kg (All Speeds - 验证基线漂移适应性)
# --------------------------------------------------------
echo ""
echo ">>> [Step 2.3] Evaluating Load Transfer (0kg)..."
$PYTHON main.py --mode eval \
    --scenario transfer_load_0kg \
    --exp_name ${EXP_NAME} \
    $COMMON_ARGS

# --------------------------------------------------------
# Step 5: 评估 - 重载迁移 (Evaluation - Load Transfer 400kg)
# Domain: 400kg (All Speeds - 验证高能基线适应性)
# --------------------------------------------------------
echo ""
echo ">>> [Step 2.4] Evaluating Load Transfer (400kg)..."
$PYTHON main.py --mode eval \
    --scenario transfer_load_400kg \
    --exp_name ${EXP_NAME} \
    $COMMON_ARGS

echo ""
echo "========================================================"
echo "   所有实验执行完毕！(All Done)"
echo "   请查看汇总报表: checkpoints/${EXP_NAME}/eval_results/summary_report.csv"
echo "   请查看可视化图: checkpoints/${EXP_NAME}/vis_pdf/"
echo "========================================================"
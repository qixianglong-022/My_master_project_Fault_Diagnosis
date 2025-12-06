#!/bin/bash

# ========================================================
#   硕士毕业论文全流程实验自动化脚本 (Final Fixed Version)
#   配置：Global Z-Score + Speed Covariate (No RevIN)
# ========================================================

# 定义实验名称前缀
EXP_PREFIX="Thesis_Final_Fixed"
SOURCE_EXP_NAME="${EXP_PREFIX}_Source_Model"

# 关键参数：强制禁用 RevIN，防止 main.py 默认开启
ABLATION_MODE="no_revin"

echo "========================================================"
echo "   开始运行全量实验..."
echo "   Mode: ${ABLATION_MODE} (Disable RevIN, Enable Speed)"
echo "   Target Dir: checkpoints/${SOURCE_EXP_NAME}"
echo "========================================================"

# --------------------------------------------------------
# 第一步：训练源域模型 (200kg)
# --------------------------------------------------------
echo ""
echo ">>> [Step 1] Training Source Model (Baseline)..."
python main.py --mode train \
    --scenario baseline \
    --ablation ${ABLATION_MODE} \
    --exp_name ${SOURCE_EXP_NAME} \
    --lr 0.001 --epochs 50 --batch_size 64

if [ $? -ne 0 ]; then
    echo "Training failed! Exiting."
    exit 1
fi

# --------------------------------------------------------
# 第二步：执行各项测试 (复用 Step 1 的模型)
# --------------------------------------------------------

# 2.1 基础实验测试 (200kg 同工况)
echo ""
echo ">>> [Step 2.1] Testing Baseline (Seen Conditions)..."
python main.py --mode eval \
    --scenario baseline \
    --ablation ${ABLATION_MODE} \
    --exp_name ${SOURCE_EXP_NAME}

# 2.2 转速迁移测试 (200kg 未见转速)
echo ""
echo ">>> [Step 2.2] Testing Speed Transfer (200kg Unseen)..."
python main.py --mode eval \
    --scenario transfer_speed \
    --ablation ${ABLATION_MODE} \
    --exp_name ${SOURCE_EXP_NAME}

# 2.3 轻载迁移测试 (0kg 全工况)
echo ""
echo ">>> [Step 2.3] Testing Load Transfer (0kg)..."
python main.py --mode eval \
    --scenario transfer_load_0kg \
    --ablation ${ABLATION_MODE} \
    --exp_name ${SOURCE_EXP_NAME}

# 2.4 重载迁移测试 (400kg 全工况)
# 这是验证 LSTMAE 崩盘、RDLinear 坚挺的关键实验
echo ""
echo ">>> [Step 2.4] Testing Load Transfer (400kg)..."
python main.py --mode eval \
    --scenario transfer_load_400kg \
    --ablation ${ABLATION_MODE} \
    --exp_name ${SOURCE_EXP_NAME}

echo ""
echo "========================================================"
echo "   所有实验完成！"
echo "   结果汇总请查看: checkpoints/${SOURCE_EXP_NAME}/eval_results/summary_report.csv"
echo "========================================================"
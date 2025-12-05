#!/bin/bash

# 定义实验名称前缀
EXP_PREFIX="Thesis_Final"
# 强力建议：先训练一个通用的 Source Model，然后用它去测所有场景
# 这样最节省时间，也能保证所有测试都是基于同一个模型权重
SOURCE_EXP_NAME="${EXP_PREFIX}_Source_Model"

echo "========================================================"
echo "   硕士毕业论文全流程实验自动化脚本"
echo "   Source Domain: 200kg (15, 45, 15-45, 45-15)"
echo "========================================================"

# --------------------------------------------------------
# 第一步：训练源域模型 (既是 Baseline 的训练，也是所有迁移实验的母模型)
# --------------------------------------------------------
echo ""
echo ">>> [Step 1] Training Source Model (Baseline)..."
# 使用 'baseline' 场景进行训练，它会保存模型到 checkpoints/Thesis_Final_Source_Model
python main.py --mode train \
    --scenario baseline \
    --exp_name ${SOURCE_EXP_NAME} \
    --lr 0.001 --epochs 50 --batch_size 64

if [ $? -ne 0 ]; then
    echo "Training failed! Exiting."
    exit 1
fi

# --------------------------------------------------------
# 第二步：执行各项测试 (复用 Step 1 的模型)
# 注意：我们使用 --mode eval，并指向同一个 --exp_name
# 但我们通过切换 --scenario 来改变测试集！
# (这就要求 main.py 在 eval 模式下也能正确加载 scenario 的 test_atoms)
# --------------------------------------------------------

# 2.1 基础实验测试 (200kg Seen)
echo ""
echo ">>> [Step 2.1] Testing Baseline (Seen Conditions)..."
python main.py --mode eval \
    --scenario baseline \
    --exp_name ${SOURCE_EXP_NAME}

# 2.2 转速迁移测试 (200kg Unseen)
echo ""
echo ">>> [Step 2.2] Testing Speed Transfer (200kg Unseen)..."
python main.py --mode eval \
    --scenario transfer_speed \
    --exp_name ${SOURCE_EXP_NAME}

# 2.3 轻载迁移测试 (0kg All)
echo ""
echo ">>> [Step 2.3] Testing Load Transfer (0kg)..."
python main.py --mode eval \
    --scenario transfer_load_0kg \
    --exp_name ${SOURCE_EXP_NAME}

# 2.4 重载迁移测试 (400kg All)
echo ""
echo ">>> [Step 2.4] Testing Load Transfer (400kg)..."
python main.py --mode eval \
    --scenario transfer_load_400kg \
    --exp_name ${SOURCE_EXP_NAME}

echo ""
echo "========================================================"
echo "   所有实验完成！"
echo "   结果汇总请查看: checkpoints/${SOURCE_EXP_NAME}/eval_results/summary_report.csv"
echo "   (注意：report.csv 可能会被最后一次运行覆盖，建议每次跑完手动备份一下 csv，或者修改 run_evaluation 代码支持追加写入)"
echo "========================================================"
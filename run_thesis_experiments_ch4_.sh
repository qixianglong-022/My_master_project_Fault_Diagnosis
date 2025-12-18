#!/bin/bash
# run_thesis_experiments_ch4.sh

# 设置 Python 路径
PYTHON="python"

echo "=================================================================="
echo "   Chapter 4: Physics-Guided Lightweight Fault Diagnosis"
echo "   Experimental Protocol: Single Source Domain Generalization"
echo "=================================================================="
echo ">>> Source Domain (Train): Load 200kg | Speeds: 15, 45, 15-45, 45-15"
echo ">>> Target Domain (Test):  Load 0/400kg | Speeds: All 8 Types"
echo "=================================================================="

# 1. 预处理 (确保数据已生成)
# -----------------------------------------------
echo ""
echo "[Step 0] Checking Data..."
if [ ! -d "processed_data_ch4_dual_stream" ]; then
    echo "Processing raw data..."
    $PYTHON preprocess_ch4_manager.py
fi

# 2. RQ1: 域泛化性能主实验 (Phys-RDLinear)
# -----------------------------------------------
echo ""
echo "[Step 1] RQ1: Running Main Experiment (Phys-RDLinear)..."
# 训练模型 (源域 200kg)
$PYTHON main_ch4.py \
    --model_name Phys-RDLinear \
    --exp_name "Ch4_RQ1_PhysRDLinear_Source200" \
    --mode train

# 评估泛化性 (目标域 0kg & 400kg)
$PYTHON main_ch4.py \
    --model_name Phys-RDLinear \
    --exp_name "Ch4_RQ1_PhysRDLinear_Source200" \
    --mode eval

# 3. RQ2: 多分辨率消融 (Multi-Resolution Ablation)
# -----------------------------------------------
# 对比: 仅使用全景流 (Single Stream) vs 双流
echo ""
echo "[Step 2] RQ2: Multi-Resolution Ablation..."
# 这里需要在代码中通过参数控制 (假设 main_ch4 支持 --ablation 参数)
# $PYTHON main_ch4.py --model_name Phys-RDLinear --ablation no_micro ...

# 4. RQ3: 物理组件消融 (Component Ablation)
# -----------------------------------------------
# 对比: 无 PGFA, 无 MTL
echo ""
echo "[Step 3] RQ3: Component Ablation..."
# $PYTHON main_ch4.py --model_name Phys-RDLinear --ablation no_pgfa ...
# $PYTHON main_ch4.py --model_name Phys-RDLinear --ablation no_mtl ...

echo ""
echo ">>> All Chapter 4 Experiments Completed."
#!/bin/bash
# run_all_baselines.sh
# 自动化完成 RQ2: 消融实验与模型对比

MODELS=("Phys-RDLinear" "FD-CNN" "TiDE" "ResNet-18")
SCENARIO="challenge_transfer"

for MODEL in "${MODELS[@]}"
do
    echo "------------------------------------------------"
    echo ">>> 开始对比实验: Model=${MODEL} | Scenario=${SCENARIO}"
    echo "------------------------------------------------"

    # 执行预处理（针对第四章双流特征）
    python preprocess_ch4_integrated.py

    # 执行训练与测试
    python main_ch4.py \
        --model_name ${MODEL} \
        --scenario ${SCENARIO} \
        --exp_name "Thesis_Final_Comparison_${MODEL}" \
        --mode all

    echo ">>> 实验 ${MODEL} 完成，结果保存在 checkpoints/Thesis_Final_Comparison_${MODEL}"
done

# 生成对比报表
python plot_fig_chapter4_comparison.py
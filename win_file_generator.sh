#!/bin/bash

# 定义根目录（默认当前目录，可修改为绝对路径，例如 D:/project 或 /c/project）
ROOT_DIR="./"

# ====================== 第一步：创建目录结构 ======================
echo "开始创建目录结构..."
mkdir -p "${ROOT_DIR}/models"  # -p 确保父目录存在，无则创建
mkdir -p "${ROOT_DIR}/utils"
echo "目录创建完成！"

# ====================== 第二步：创建空文件 ======================
echo "开始创建文件..."

# 根目录文件
touch "${ROOT_DIR}/config.py"
touch "${ROOT_DIR}/data_loader.py"
touch "${ROOT_DIR}/train.py"
touch "${ROOT_DIR}/deploy.py"
touch "${ROOT_DIR}/requirements.txt"

# models 目录文件
touch "${ROOT_DIR}/models/__init__.py"
touch "${ROOT_DIR}/models/rdlinear.py"
touch "${ROOT_DIR}/models/layers.py"

# utils 目录文件
touch "${ROOT_DIR}/utils/__init__.py"
touch "${ROOT_DIR}/utils/anomaly.py"
touch "${ROOT_DIR}/utils/metrics.py"
touch "${ROOT_DIR}/utils/tools.py"

echo "所有文件创建完成！"

# ====================== 可选：添加文件头部注释（增强版） ======================
read -p "是否为文件添加基础头部注释？(y/n) " -n 1 -r
echo    # 换行
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "开始添加文件头部注释..."
    
    # 定义通用注释模板
    COMMENT_TEMPLATE=$(cat << 'EOF'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : {FILENAME}
@Author  : Your Name
@Date    : $(date +%Y-%m-%d)
@Desc    : {DESC}
"""

EOF
    )

    # 为每个文件写入注释（按需求自定义描述）
    # config.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/config.py//\{DESC\}/所有的超参数（窗口大小、路径、学习率）都在这里配置，替代复杂的 argparse}" > "${ROOT_DIR}/config.py"
    
    # data_loader.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/data_loader.py//\{DESC\}/负责加载 CSV，处理转速平方，生成 PyTorch Dataset}" > "${ROOT_DIR}/data_loader.py"
    
    # models/rdlinear.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/rdlinear.py//\{DESC\}/实现 RevIN + 转速引导的 Trend 分支（核心）}" > "${ROOT_DIR}/models/rdlinear.py"
    
    # models/layers.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/layers.py//\{DESC\}/存放 RevIN 和 MovingAverage 的基础实现}" > "${ROOT_DIR}/models/layers.py"
    
    # utils/anomaly.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/anomaly.py//\{DESC\}/原 anomaly_utils.py，负责 SPE 和 POT 阈值}" > "${ROOT_DIR}/utils/anomaly.py"
    
    # utils/metrics.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/metrics.py//\{DESC\}/负责计算 Precision, Recall, F1, AUC}" > "${ROOT_DIR}/utils/metrics.py"
    
    # utils/tools.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/tools.py//\{DESC\}/EarlyStopping, Logger}" > "${ROOT_DIR}/utils/tools.py"
    
    # train.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/train.py//\{DESC\}/PC 端训练脚本，生成 .pth 和 .onnx}" > "${ROOT_DIR}/train.py"
    
    # deploy.py
    echo "${COMMENT_TEMPLATE//\{FILENAME\}/deploy.py//\{DESC\}/树莓派端推理脚本，加载 .onnx，实时输出异常得分}" > "${ROOT_DIR}/deploy.py"
    
    # requirements.txt（单独处理，无 Python 注释）
    echo "# 依赖库列表
torch>=2.0.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.2.0
onnx>=1.14.0
onnxruntime>=1.15.0" > "${ROOT_DIR}/requirements.txt"

    echo "头部注释添加完成！"
fi

# 输出最终目录结构（验证）
echo -e "\n生成的目录结构如下："
tree "${ROOT_DIR}" --dirsfirst 2>/dev/null || ls -R "${ROOT_DIR}"  # 兼容无 tree 命令的环境
echo -e "\n操作完成！文件生成路径：${ROOT_DIR}"
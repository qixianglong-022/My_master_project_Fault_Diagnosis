这一份总结将是你写论文第三章（或第四章）实验部分的**核心骨架**。你可以直接把这部分内容扩充成论文的章节。

以下是针对你目前工作的全方位总结：

---

### 1. 实验逻辑与分组设计 (Experimental Design)

**核心目标：** 探究多源信息融合在故障诊断中的有效性，并验证基于“预测误差（Reconstruction Error）”的**半监督**异常检测方法的适用性。

**至少需要完成的三组对比实验：**

* **实验组 A（单一振动基准）：**
    * **输入：** 仅使用振动原始信号（如电机振动）。
    * **目的：** 建立振动信号检测故障的基准性能（Baseline）。
* **实验组 B（单一声音基准）：**
    * **输入：** 仅使用声纹/麦克风信号。
    * **目的：** 探究声音信号在特定故障（如早期微弱故障或气动噪声）下的敏感度。
* **实验组 C（多源数据融合）：**
    * **输入：** 振动 + 声音信号（特征级拼接 或 数据级通道叠加）。
    * **目的：** 验证融合是否能结合两者优势，提升 F1-Score，或分析在什么情况下融合反而引入了噪声。

注：实验初期只考虑恒定工况，目前选取**正常工况HH-0-3**和**轴承故障工况B-F-0-3**

---

### 2. 数据集生成逻辑 (Data Preparation)

**原则：** 训练集必须纯净（只含正常），测试集必须全面（含正常+故障）。

* **训练/验证集生成（Training/Validation Set）：**
    * **来源：** 纯正常工况数据（例如 `HH-0-3.csv`）。
    * **选取：**使用`data_process.py`脚本，生成`HH-0-3_Motor_Vibration_train.csv`
    * **切分：** 代码自动将文件前 70% 划为训练集，用于训练 DLinear；中间 10% 划为验证集，**用于计算 POT 动态阈值**。
    * **预处理：** 使用 `StandardScaler` 进行标准化，**并保存该 Scaler (`scaler.pkl`)**。
    
* **测试集生成（Test Set）：**
    * **来源：** 手动`HH-0-3_Motor_Vibration_train.csv`和`BF-0-3_Motor_Vibration_train.csv`拼接文件（例如 `Mixed_HH_FB_0-3_Motor_Vibration_test.csv`）。
    * **结构：** `[ 一段正常数据50% | 一段故障数据50% ]`。
    * **标签：** 配套生成的 `Mixed_HH_FB_0-3_Motor_Vibration_test.npy` 标签文件，正常段标记为 0，故障段标记为 1。
    * **关键点：** **必须复用训练集的 Scaler** 进行标准化，严禁在测试集上重新计算均值方差，以防故障特征被抹除。

---

### 3. 训练程序逻辑 (`train_gpu.py`)

**核心思想：** 甚至不需要见过故障，只需要把“正常”学得滚瓜烂熟。

1.  **加载数据：** 读取正常数据，拟合 Scaler。
2.  **模型训练 (Model Training)：**
    * **模型：** DLinear (分解为趋势项 + 季节项)。
    * **任务：** 时序预测（用过去 96 个点预测未来 96 个点）。
    * **Loss：** MSE（均方误差），迫使模型极度适应正常波形。
    * **产出 1：** 最佳模型权重 (`DLinear_best.pth`)。
    * **产出 2：** 数据标准化器 (`scaler.pkl`)。
3.  **阈值标定 (Threshold Calibration)：**
    * 使用验证集（Validation Set）跑一遍模型。
    * 计算验证集的 SPE（预测误差）。`test_spe_Motot_Vibration.npy`
    * **POT 算法：** 利用极值理论分析 SPE 的尾部分布，在给定风险概率 `q` 下自动计算出 **动态阈值 (`test_threshold_Motot_Vibration.npy`)**。

---

### 4. 测试程序逻辑 (`inference_anomaly.py`)

**核心思想：** 用“正常”的标准去衡量一切，误差大的就是“异常”。

1.  **加载资产：**
    * 加载模型权重 (`DLinear_best.pth`)。
    * 加载**训练时的** Scaler (`scaler.pkl`)。
    * 加载**训练时的** 阈值 (`test_threshold_Motot_Vibration.npy`)。
2.  **全量推理：**
    * 读取混合测试集（`flag='all'` 模式）。
    * DLinear 进行预测，得到预测值。
3.  **异常判定：**
    * 计算 **SPE (Squared Prediction Error)**：$(真实值 - 预测值)^2$。
    * **判据：**
        * 若 `SPE > Threshold` $\rightarrow$ 判定为 **1 (故障)**。
        * 若 `SPE <= Threshold` $\rightarrow$ 判定为 **0 (正常)**。
4.  **指标评估：**
    * 与真实标签 (`gt_labels`) 对比。
    * 计算 Accuracy, Precision (查准), Recall (查全), F1-Score。
    * 生成混淆矩阵。
    * 生成`final_inference_labels_Moter_Vibration.npy`和`final_inference_spe_Moter_Vibration.npy`。

---

### 5. 论文所需生成的图片 (Figures)

你需要为每一组实验（振动、声音、融合）生成以下图片：

1.  **异常得分曲线图 (The Anomaly Score Plot) —— **最核心的图****
    * **横轴：** 时间 (Time/Sample Index)。
    * **纵轴：** SPE 值 (Anomaly Score)。
    * **内容：**
        * 绘制测试集的 SPE 曲线（通常是蓝线）。
        * 画一条红色的水平线表示 **阈值 (Threshold)**。
        * 用背景阴影或不同颜色标出 **真实故障区域 (Ground Truth)**。
    * **预期效果：** 正常区域蓝线在红线下方，进入故障区域后蓝线瞬间飙升超过红线。

2.  **原始信号与预测信号对比图 (Raw vs. Prediction)**
    * 选一段包含“正常转故障”时刻的数据。
    * 画出 `Input` (真实值) 和 `Output` (预测值)。
    * **预期效果：** 正常阶段两条线重合度高；故障阶段两条线分离（模型试图用正常逻辑去预测故障，导致预测偏离）。

3.  **混淆矩阵热力图 (Confusion Matrix Heatmap)**
    * 直观展示 TP, TN, FP, FN。

4.  **多源对比柱状图 (Performance Comparison)**
    * 把振动、声音、融合三组实验的 F1-Score 画成柱状图，一目了然地展示谁效果好。

---

### 6. 如何从“单一振动”切换为“单一声音” (Step-by-Step)

当你跑完振动实验后，要切到声音实验，只需按以下步骤操作：

1.  **数据准备：**
    * 准备好声音的正常数据 CSV（如 `Sound_Normal.csv`）。
    * 准备好声音的混合测试数据 CSV（如 `Sound_Mixed_Test.csv`）和对应标签 `.npy`。
2.  **修改 `train_gpu.py` 参数：**
    * `--data_path`: 改为 `Sound_Normal.csv`。
    * `--target`: 改为声音数据的列名（如果有特定目标列）。
    * **`--output_dir` (至关重要):** 改为 `./output_sound/`。**千万别忘了改这个，否则会覆盖掉你刚才跑的振动模型！**
3.  **运行训练：** 跑 `train_gpu.py`，生成新的模型、Scaler 和 阈值。
4.  **修改 `inference_anomaly.py` 参数：**
    * `--output_dir`: 指向 `./output_sound/`。
    * `--data_path`: 改为 `Sound_Mixed_Test.csv`。
    * 代码里加载标签的路径改为声音的标签文件。
5.  **运行测试：** 跑 `inference_anomaly.py`，记录结果。

**总结完毕！** 这就是你这章论文完整的工程实现路径。
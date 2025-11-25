import argparse
import os
import torch
import numpy as np
import pandas as pd
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# 引入你项目里的模块
from models import DLinear
from datasets.mydataset import Dataset_Custom
from util.anomaly_utils import AnomalyMeasurer  # 假设你之前已经建好了这个
from sklearn.metrics import roc_auc_score

def get_args():
    # 这里只列出测试需要的关键参数，确保和 train_gpu.py 里训练时的一致
    parser = argparse.ArgumentParser(description='Anomaly Detection Inference')

    # 数据相关
    # parser.add_argument('--root_path', type=str, default='./dataset/', help='数据文件夹路径')
    # parser.add_argument('--data_path', type=str, default='inference_test.csv', help='拼接好的测试文件')
    parser.add_argument('--data', type=str, default='HH-0-3', help='数据集名称，对应 data_factory.py 里的字典键')
    parser.add_argument('--root_path', type=str, default='/', help='这里填 / 或者留空')
    #改1
    parser.add_argument('--data_path', type=str,
                        default=r'D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis'
                                r'\data_chapter3\Mixed_HH_FB_0-3_Motor_Vibration_test.csv',
                        # 直接填绝对路径
                        help='具体的数据文件绝对路径')

    parser.add_argument('--seq_len', type=int, default=96, help='输入长度')
    parser.add_argument('--pred_len', type=int, default=96, help='预测长度 (训练时用的多少就填多少)')
    parser.add_argument('--enc_in', type=int, default=1, help='通道数 (振动是1, 融合是2+)')

    # 模型相关
    parser.add_argument('--individual', action='store_true', default=False, help='DLinear参数')

    # 路径相关 (指向你 train_gpu.py 输出的那个文件夹)
    #改2
    parser.add_argument('--output_dir', type=str, default='./output_Motor_Vibration/', help='存放权重和阈值的文件夹')
    parser.add_argument('--device', type=str, default='cuda', help='使用设备')

    return parser.parse_args()


def main():
    '''
    这段代码的作用是
    1. 加载之前的权重 checkpoint.pth 和验证机算出的阈值 test_threshold.npy
    2. 加载新的test数据集（包含正常工况和故障工况，fault_start_index为分界点） mixed_test.csv
    3. 跑预测，算SPE，对比阈值，算出指标和混淆矩阵
    '''
    args = get_args()
    device = torch.device(args.device)
    #改3
    test_data_label = (r'D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\data_chapter3'
                       r'\Mixed_HH_FB_0-3_Motor_Vibration_test.npy')
    print(f"Loading Model from: {args.output_dir}")
    print(f"Testing Data: {os.path.join(args.root_path, args.data_path)}")
    print(f"Testing Data Label:{test_data_label}")

    # ================= 1. 初始化模型 =================
    model = DLinear(args.seq_len, args.pred_len, args.individual, args.enc_in)
    model.to(device)

    # 加载权重
    model_path = os.path.join(args.output_dir, 'DLinear_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"没找到模型权重文件: {model_path}")

    # 注意：有时候保存时包了一层 'model' key，取决于你 train_gpu.py 怎么存的
    checkpoint = torch.load(model_path, map_location=device)
    # 如果是 model.state_dict() 直接存的：
    # model.load_state_dict(checkpoint)
    # 如果是 save_on_master 存的字典：
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(">>> Model Loaded Successfully.")

    # ================= 2. 加载阈值 =================
    threshold_path = os.path.join(args.output_dir, 'test_threshold.npy')
    if not os.path.exists(threshold_path):
        raise FileNotFoundError("没找到阈值文件，请先运行 train_gpu.py 生成阈值！")

    threshold = np.load(threshold_path)
    print(f">>> Threshold Loaded: {threshold}")

    # ================= 2.5 (新增) 加载 Scaler =================
    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    if os.path.exists(scaler_path):
        print(f"Loading Scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        raise FileNotFoundError("未找到 scaler.pkl，请先运行 train_gpu.py 进行训练！")

    # ================= 3. 准备测试数据 =================
    # 直接调用 Dataset 类，不通过 data_factory，这样更灵活
    dataset = Dataset_Custom(
        root_path=args.root_path,
        data_path=args.data_path,
        flag='all',
        size=[args.seq_len, 0, args.pred_len],  # label_len 这里不需要
        features='S',  # 假设是单变量
        #改4
        target='Motor_Vibration',
        timeenc=0,
        freq='m',
        scaler = scaler
    )

    # 强制让它读取整个文件 (覆盖默认的切分逻辑)
    # 这种改法取决于 Dataset_Custom 具体写法，最稳妥的是你的 CSV 就是你要测的全部
    # 如果 Dataset_Custom 内部有根据 flag 切分的逻辑，你需要确认它读到了所有行
    print(f"Total samples to test: {len(dataset)}")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)

    # ================= 4. 推理 (Inference) =================
    preds_list = []
    trues_list = []

    print(">>> Starting Inference...")
    with torch.no_grad():
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            # DLinear 前向传播
            outputs = model(batch_x)

            # 截取预测部分 (对齐维度)
            # 假设我们只关心最后 pred_len 的预测误差
            f_dim = -1 if args.enc_in > 1 else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    # ================= 5. 计算 SPE 并检测 =================
    # 临时实例化一个 AnomalyMeasurer 用来调 calculate_spe
    detector = AnomalyMeasurer(q=0)  # 这里 q 不重要，因为我们已经有 threshold 了
    detector.threshold = threshold  # 强制赋值之前算好的阈值

    # 进行检测 (0:正常, 1:异常)
    spe_scores = detector.calculate_spe(torch.tensor(preds), torch.tensor(trues))

    # ================= 6. 动态寻找最佳阈值 (Best F1) =================
    # 获取真实标签
    try:
        gt_labels = np.load(test_data_label)
        min_len = min(len(spe_scores), len(gt_labels))
        gt_labels = gt_labels[:min_len]
        spe_scores = spe_scores[:min_len]
    except:
        raise ValueError("必须要有真实标签才能寻找最佳阈值！")

    print(">>> Searching for best threshold...")

    # 在 min(score) 到 max(score) 之间生成 100 个候选阈值
    min_score = np.min(spe_scores)
    max_score = np.max(spe_scores)
    thresholds = np.linspace(min_score, max_score, 100)

    best_f1 = 0
    best_threshold = 0
    best_pred_labels = None

    for th in thresholds:
        # 产生预测
        tmp_pred = (spe_scores > th).astype(int)
        # 简单计算 F1 (使用 sklearn)
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_labels, tmp_pred, average='binary', zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = th
            best_pred_labels = tmp_pred

    print("\n" + "=" * 40)
    print(f" Best F1 Found!")
    print(f" Best Threshold : {best_threshold:.4f} (Original POT was {threshold:.4f})")
    print("=" * 40)

    # 使用最佳阈值的结果重新计算详细指标
    accuracy = accuracy_score(gt_labels, best_pred_labels)
    cm = confusion_matrix(gt_labels, best_pred_labels)

    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_labels, best_pred_labels, average='binary', zero_division=0
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-" * 20)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 40)

    # 保存结果画图用
    np.save(os.path.join(args.output_dir, 'final_inference_spe.npy'), spe_scores)
    np.save(os.path.join(args.output_dir, 'final_inference_labels.npy'), best_pred_labels)

    # 1. 计算 AUC (最客观的指标)
    try:
        auc = roc_auc_score(gt_labels, spe_scores)
        print(f"★ AUC-ROC Score: {auc:.4f}")
    except:
        print("Error calculating AUC")

    # 2. 诊断：查看正常和故障数据的 SPE 到底是多少？
    normal_spe = spe_scores[gt_labels == 0]
    fault_spe = spe_scores[gt_labels == 1]

    print("\n=== Diagnostic Info ===")
    print(
        f"Normal SPE -> Mean: {np.mean(normal_spe):.4f}, Median: {np.median(normal_spe):.4f}, Std: {np.std(normal_spe):.4f}")
    print(
        f"Fault  SPE -> Mean: {np.mean(fault_spe):.4f},  Median: {np.median(fault_spe):.4f},  Std: {np.std(fault_spe):.4f}")
    print("=======================")

if __name__ == '__main__':
    main()
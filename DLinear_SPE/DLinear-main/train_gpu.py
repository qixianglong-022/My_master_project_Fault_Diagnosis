""" ImageNet Training Script (基于 DLinear 修改版)
这个脚本是整个训练、验证和测试的入口。
它负责：
1. 解析命令行参数 (Epochs, Batch_size, 数据路径等)
2. 加载数据 (Dataset & DataLoader)
3. 初始化模型 (DLinear)
4. 定义优化器和损失函数
5. 执行训练循环 (Train -> Validate -> Save Checkpoint)
6. [新增] 执行异常检测评估 (SPE + POT)
"""
import argparse
import datetime
import numpy as np
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import json
import os
import joblib
from pathlib import Path
from timm.utils import NativeScaler, get_state_dict, ModelEma
from util.anomaly_utils import AnomalyMeasurer  # [核心工具] 引入我们写的异常检测工具类

from util.samplers import RASampler
from util import utils as utils
from util.optimizer import Lion
from util.loss import loss_family
from util.engine import train_one_epoch, evaluate
from util.lr_sched import create_lr_scheduler

from datasets import build_dataset
from models import DLinear  # 我们主要关注 DLinear


def get_args_parser():
    parser = argparse.ArgumentParser('DLinear Training and Evaluation Script', add_help=False)

    # =========================
    # 1. 基础训练参数 (Basic Training)
    # =========================
    parser.add_argument('--batch_size', default=128, type=int, help='批大小')
    parser.add_argument('--epochs', default=10, type=int, help='训练轮数')
    parser.add_argument('--loss_type', default='MAE', type=str,
                        choices=['MAE', 'MSE', 'RMSE', 'MAPE', 'MSPE', 'RSE', 'CORR'],
                        help='损失函数类型，推荐 MSE')

    # =========================
    # 2. 数据集参数 (Data)
    # =========================
    # parser.add_argument('--data', type=str, default='custom', help='数据集类型')
    # parser.add_argument('--root_path', type=str, default='./TimeseriesData', help='数据根目录')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='S',
                        help='预测任务: [M]多变量预测多变量, [S]单变量预测单变量, [MS]多变量预测单变量')
    # 改1
    parser.add_argument('--target', type=str, default='Motor_Vibration', help='目标列名(S或MS任务用)')
    parser.add_argument('--freq', type=str, default='m',
                        help='时间频率(s/t/h/d/b/w/m)，51.2k数据填h即可')
    parser.add_argument('--num_workers', default=0, type=int, help='数据加载线程数，Windows建议0，Linux可设为4或8')
    parser.add_argument('--pin-mem', action='store_true',
                        help='开启页锁定内存，加速CPU到GPU的数据传输')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)
    # --- 数据集参数 (重点) ---

    #改2
    parser.add_argument('--data', type=str, default='HH-0-3', help='数据集名称，对应 data_factory.py 里的字典键')
    parser.add_argument('--root_path', type=str, default='/', help='这里填 / 或者留空')
    parser.add_argument('--data_path', type=str,

                        default=r'D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\data_chapter3\HH-0-3_Motor_Vibration_train.csv',help='具体的数据文件绝对路径')
                        # 直接填绝对路径

    # =========================
    # 3. 模型参数 (Model Structure)
    # =========================
    parser.add_argument('--model', default='DLinear', type=str, help='模型名称')
    parser.add_argument('--seq_len', type=int, default=1024, help='输入序列长度') # 锁死输入序列长度
    parser.add_argument('--label_len', type=int, default=48, help='Informer类模型所需的标签长度')
    parser.add_argument('--pred_len', type=int, default=1, help='预测序列长度') # 锁死预测序列长度

    # --- DLinear 核心参数 ---
    parser.add_argument('--individual', action='store_true', default=True,
                        help='DLinear特有: 是否对每个通道单独建模(True)还是共享权重(False)')
    parser.add_argument('--enc_in', type=int, default=1, help='输入通道数 (振动=1, 融合=2等)')

    # --- Transformer 类通用参数 (DLinear 部分不使用，但保留以兼容代码接口) ---
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入通道数')
    parser.add_argument('--c_out', type=int, default=7, help='输出通道数')
    parser.add_argument('--d_model', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--d_ff', type=int, default=2048, help='全连接层维度')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--factor', type=int, default=1, help='注意力因子')
    parser.add_argument('--distil', action='store_false', default=True, help='是否在编码器中使用蒸馏')
    parser.add_argument('--dropout', type=float, default=0.05, help='Dropout比率')
    parser.add_argument('--embed', type=str, default='timeF', help='时间编码方式')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='是否输出注意力权重')
    parser.add_argument('--embed_type', type=int, default=0, help='嵌入类型')

    # =========================
    # 4. 优化与正则化 (Optimization & EMA)
    # =========================
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')

    # --- 梯度裁剪 (防止梯度爆炸) ---
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='梯度裁剪范数 (默认None，推荐 0.5-5.0)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='梯度裁剪模式: "norm", "value", "agc"')

    # --- EMA (指数移动平均，提升模型鲁棒性) ---
    parser.add_argument('--model-ema', action='store_true', help='开启 EMA')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=False)  # 默认关闭，如果需要更稳健的结果可以开启
    parser.add_argument('--model-ema-decay', type=float, default=0.9999, help='EMA 衰减率')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='强制在CPU上更新EMA')

    # =========================
    # 5. 系统与IO (System & IO)
    # =========================
    # 改3
    parser.add_argument('--output_dir', default='./output_Motor_Vibration', help='模型保存路径')
    parser.add_argument('--writer_output', default='./runs', help='Tensorboard日志路径')
    parser.add_argument('--device', default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--seed', default=0, type=int, help='随机种子')
    parser.add_argument('--save_freq', default=1, type=int, help='每隔多少个epoch保存一次模型')

    # --- 断点续训 (Resume) ---
    parser.add_argument('--resume', default='', help='检查点路径 (例如: output/checkpoint.pth)，用于恢复训练')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='手动设置开始的epoch')

    # --- 评估模式 ---
    parser.add_argument('--eval', action='store_true', help='仅执行评估，不训练')
    parser.add_argument('--train_only', action='store_true', help='仅在完整数据上训练，不验证不测试')
    parser.add_argument('--set_ln_eval', action='store_true', default=False, help='微调时将BN层设为eval模式')

    # =========================
    # 6. 分布式训练 (Distributed - 单卡可忽略)
    # =========================
    parser.add_argument('--world_size', default=1, type=int, help='分布式进程数')
    parser.add_argument('--local_rank', default=0, type=int, help='本地GPU编号')
    parser.add_argument('--dist_url', default='env://', help='分布式初始化URL')
    parser.add_argument('--dist-eval', action='store_true', default=False, help='开启分布式评估')

    return parser

def main(args):
    print(args)
    # 初始化分布式环境 (单卡会自动跳过)
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # --- 1. 设置随机种子 (重要：写论文必须固定种子) ---
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- 2. 构建数据集 ---
    # build_dataset 会去调用 mydataset.py 里的逻辑
    # flag='train' 表示加载训练集 (通常只包含正常数据)
    dataset_train, _ = build_dataset(args=args, flag='train')
    # ================= 新增保存逻辑 =================
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    scaler_path = os.path.join(args.output_dir, 'scaler.pkl')
    print(f"Saving scaler to {scaler_path} ...")
    # 从 dataset_train 中提取出刚刚 fit 好的 scaler 并保存
    joblib.dump(dataset_train.scaler, scaler_path)
    # ===============================================
    dataset_val, _ = build_dataset(args=args, flag='val')

    # 创建采样器 (Sampler) 和 数据加载器 (DataLoader)
    # DataLoader 负责把数据打包成 Batch
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True, # 丢弃最后不足一个batch的数据
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # --- 3. 初始化模型 ---
    print(f"Creating model: {args.model}")
    if args.model == 'DLinear':
        # DLinear 初始化，传入序列长度、预测长度、是否独立通道、通道数
        model = DLinear(args.seq_len, args.pred_len, args.individual, args.enc_in)
    else:
        # 你的论文主要用 DLinear，这里为了兼容代码可以保留，但建议直接用上面的
        model = DLinear(args.seq_len, args.pred_len, args.individual, args.enc_in)

    model.to(device)

    # 计算模型参数量，写论文时可以用来证明模型很轻量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    # --- 4. 定义优化器和损失函数 ---
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # 学习率调度器 (Warmup + Cosine Decay)
    lr_scheduler = create_lr_scheduler(optimizer, num_step=len(data_loader_train), epochs=args.epochs, warmup=True)

    print(f'Create loss calculation: {args.loss_type}')
    criterion = loss_family[args.loss_type] # 获取损失函数 (通常是 MSELoss)
    best_loss = 100.0

    # --- 5. 开始训练循环 ---
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        # A. 训练阶段
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler=None,
            # clip_mode=args.clip_grad,
            model_ema=None,
            set_training_mode=True, writer=None,
            lr_scheduler=lr_scheduler, args=args
        )

        # B. 验证阶段
        test_stats = evaluate(data_loader_val, model, criterion, device, epoch, None, args, visualization=False)
        print(f"Loss of the network on the {len(dataset_val)} test text-data: {test_stats['valid_loss']:.4f}")

        # C. 保存最佳模型
        if test_stats["valid_loss"] < best_loss:
            best_loss = test_stats["valid_loss"]
            if args.output_dir:
                # 拼接保存路径
                ckpt_path = os.path.join(args.output_dir, f'{args.model}_best.pth')
                print(f"Saving best checkpoint to {ckpt_path}")
                utils.save_on_master({
                    'model': model.state_dict(), # 只保存权重，不保存整个对象
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'best_score': best_loss,
                    'args': args,
                }, ckpt_path)

        print(f'Min {args.loss_type} loss: {best_loss:.4f}')

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

    # =======================================================
    # --- 6. 异常检测评估流程 (DLinear + SPE + POT) ---
    # =======================================================
    # =======================================================
    # 新增：基于 DLinear + SPE + POT 的异常检测流程
    # =======================================================
    print("\n-------------------------------------------------------")
    print("Starting Anomaly Detection Evaluation (DLinear + POT)")

    # 1. 实例化异常检测器
    # q=1e-4 意味着我们对正常数据的容忍度很高，只有极值才算异常
    detector = AnomalyMeasurer(q=1e-5, level=0.98)

    # 2. 准备数据：我们需要获取模型在 验证集 和 测试集 上的预测结果
    # 定义一个辅助函数来获取预测值和真实值
    def get_predictions(dataloader, model):
        model.eval()
        preds_list = []
        trues_list = []

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(dataloader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)

                # DLinear 的 forward 逻辑
                # 注意：如果是异常检测，通常是用“预测未来”或“重构当前”的误差
                # 这里假设你沿用 forecasting 任务，比较 forecast vs ground_truth
                if args.model == 'DLinear':
                    outputs = model(batch_x)
                else:
                    # 兼容其他模型接口
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # DLinear 输出通常不需要 f_dim 切片，但为了保险起见对齐维度
                f_dim = -1 if args.features == 'MS' else 0
                outputs = outputs[:, -args.pred_len:, f_dim:]
                batch_y = batch_y[:, -args.pred_len:, f_dim:]

                preds_list.append(outputs.detach().cpu())
                trues_list.append(batch_y.detach().cpu())

        preds = torch.cat(preds_list, dim=0).numpy()
        trues = torch.cat(trues_list, dim=0).numpy()
        return preds, trues

    # 3. 获取验证集数据（通常作为“正常数据”的参考来计算阈值）
    # 注意：确保你的验证集里主要包含正常样本，或者脏数据比例极低
    print("Calculating SPE on Validation Set (Normal Baseline)...")
    val_preds, val_trues = get_predictions(data_loader_val, model)
    val_spe = detector.calculate_spe(torch.tensor(val_preds), torch.tensor(val_trues))

    # 4. 拟合 POT 阈值
    # 这步是关键：根据 DLinear 对正常数据的拟合误差分布，自动画出“红线”
    dynamic_threshold = detector.fit_pot(val_spe)

    # 5. 在测试集上进行检测
    # 假设 data_loader_test 已经存在 (你代码里可能需要 build_dataset 这里的 flag='test')
    dataset_test, _ = build_dataset(args=args, flag='test')
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    print("Calculating SPE on Test Set...")
    test_preds, test_trues = get_predictions(data_loader_test, model)
    test_spe = detector.calculate_spe(torch.tensor(test_preds), torch.tensor(test_trues))

    # 6. 生成检测结果 (0: 正常, 1: 异常)
    # 在 detect 之前对 spe_scores 进行平滑
    test_spe = smooth(test_spe, window_len=10)  # 窗口越大越平滑，抗干扰越强
    pred_labels, _ = detector.detect(test_spe)

    # 7. 保存或打印结果
    print(f"Test Set Anomaly Count: {np.sum(pred_labels)} / {len(pred_labels)}")

    # 如果你有测试集的真实异常标签 (ground truth labels)，可以在这里计算 Precision/Recall/F1
    # 示例：
    # from sklearn.metrics import classification_report
    # print(classification_report(test_ground_truth_labels, pred_labels))

    # 保存 SPE 结果用于论文画图 (重要！)
    np.save(os.path.join(args.output_dir, 'test_spe.npy'), test_spe)
    np.save(os.path.join(args.output_dir, 'test_threshold.npy'), dynamic_threshold)
    print("-------------------------------------------------------")

# 在 detect 之前对 spe_scores 进行平滑
def smooth(x, window_len=5):
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.ones(window_len,'d')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser('DLinear Training', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
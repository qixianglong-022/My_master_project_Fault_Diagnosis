# main_ch4.py
import torch
import argparse
import os
from torch.utils.data import DataLoader

# 引入合并后的配置
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
# 确保这里引入的是重构后的模型文件
from models.phys_rdlinear import PhysRDLinearCls
from utils.uncertainty_loss import UncertaintyLoss
from utils.tools import set_seed
from trainer import Trainer
from utils.visualization import run_visualization_pipeline


def run_ch4_experiment():
    # 1. 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])
    args = parser.parse_args()

    # 2. 初始化配置 (直接实例化)
    config = Ch4Config()

    # 3. 环境与随机种子
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"==================================================")
    print(f">>> [Chapter 4] Experiment: Single Source DG")
    print(f"    Device: {device}")
    print(f"    TRAIN (Source): Load {config.TRAIN_LOADS} | Speeds: {len(config.TRAIN_SPEEDS)} types")
    print(f"    TEST  (Target): Load {config.TEST_LOADS} | Speeds: {len(config.TEST_SPEEDS)} types (Extrapolated)")
    print(f"==================================================")

    # 4. 准备数据
    # 现在 Ch4DualStreamDataset 可以正确接收 config 参数了
    train_ds = Ch4DualStreamDataset(config, mode='train')
    test_ds = Ch4DualStreamDataset(config, mode='test')

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f">>> Samples: Train={len(train_ds)}, Test={len(test_ds)}")

    # 5. 模型与训练器
    model = PhysRDLinearCls(config).to(device)
    trainer = Trainer(config, model, device)

    # 6. 执行流程
    if args.mode == 'train':
        print(">>> Starting Training...")
        best_acc = 0.0
        for epoch in range(config.EPOCHS):
            # 训练一个 epoch
            metrics = trainer.train_epoch(train_loader)

            # 验证泛化能力 (在未见过的 0kg/400kg 上测试)
            if (epoch + 1) % 5 == 0:
                val_acc = trainer.evaluate(test_loader)
                weights = trainer.criterion.get_weights()

                print(f"Epoch {epoch + 1}/{config.EPOCHS} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Source Acc: {metrics['acc']:.2f}% | "
                      f"Target Domain Acc: {val_acc:.2f}%")

                if val_acc > best_acc:
                    best_acc = val_acc
                    trainer.save("best_model.pth")

    elif args.mode == 'eval':

        print(">>> Starting Evaluation & Visualization...")

        path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

        if os.path.exists(path):

            # 加载模型

            model.load_state_dict(torch.load(path, map_location=device))

            # 计算目标域精度

            acc = trainer.evaluate(test_loader)

            print(f"Final Target Domain Accuracy: {acc:.2f}%")

            # ================= [新逻辑] =================

            # 为了画出“红色点”，我们需要源域数据

            # 这里的 train_loader 就是源域 (200kg)

            # 为了避免点太多图太乱，我们可以只取一部分源域数据，或者全部取

            vis_dir = os.path.join(config.CHECKPOINT_DIR, "visualizations")

            # 调用可视化流水线，传入 train_loader 作为 source

            run_visualization_pipeline(

                model,

                target_loader=test_loader,

                source_loader=train_loader,  # <--- 关键：注入源域数据

                device=device,

                output_dir=vis_dir,

                acc=acc

            )

            # ============================================


        else:

            print(f"[Error] No checkpoint found at {path}")

if __name__ == "__main__":
    run_ch4_experiment()
import os
import torch
import pandas as pd
import numpy as np
import argparse
import json
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed

# 故障类别映射 (用于后续统计特定故障的精度)
FAULT_MAP = {
    0: 'HH', 1: 'RU', 2: 'RM', 3: 'SW',
    4: 'VU', 5: 'BR', 6: 'KA', 7: 'FB'
}


class AblationTrainer(Trainer):
    """
    继承 Trainer，增加致盲功能
    """

    def __init__(self, config, model, device, ablation_mode='none'):
        super().__init__(config, model, device)
        self.ablation_mode = ablation_mode  # 'none', 'no_micro', 'no_macro'

    def _mask_input(self, micro, macro):
        """核心致盲逻辑"""
        if self.ablation_mode == 'no_micro':
            micro = torch.zeros_like(micro)
        elif self.ablation_mode == 'no_macro':
            macro = torch.zeros_like(macro)
        return micro, macro

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        for micro_x, macro_x, speed, y_cls, y_load in dataloader:
            micro_x, macro_x = micro_x.to(self.device), macro_x.to(self.device)
            speed, y_cls, y_load = speed.to(self.device), y_cls.to(self.device), y_load.to(self.device)

            # --- 致盲 ---
            micro_x, macro_x = self._mask_input(micro_x, macro_x)

            self.optimizer.zero_grad()
            # Phys-RDLinear 接受双流输入
            logits, pred_load = self.model(micro_x, macro_x, speed)
            loss, _, _ = self.criterion(logits, y_cls, pred_load, y_load)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

        return {"loss": total_loss / len(dataloader), "acc": 100 * correct / total}

    @torch.no_grad()
    def evaluate(self, dataloader, save_path=None):
        self.model.eval()
        correct, total = 0, 0
        results = []

        for micro_x, macro_x, speed, y_cls, y_load in dataloader:
            micro_x, macro_x = micro_x.to(self.device), macro_x.to(self.device)
            speed, y_cls = speed.to(self.device), y_cls.to(self.device)

            # --- 致盲 ---
            micro_x, macro_x = self._mask_input(micro_x, macro_x)

            logits, _ = self.model(micro_x, macro_x, speed)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            # 收集详细结果用于分析
            b_true = y_cls.cpu().numpy()
            b_pred = preds.cpu().numpy()
            for i in range(len(b_true)):
                results.append({
                    'True_Label': b_true[i],
                    'Pred_Label': b_pred[i],
                    'Is_Correct': 1 if b_true[i] == b_pred[i] else 0
                })

        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)

        return 100 * correct / total


def run_ablation_task(mode_name, gpu_id):
    """运行单个消融实验任务"""
    # 1. 配置
    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # 始终使用我们的模型
    # 区分 Checkpoint 目录
    config.CHECKPOINT_DIR = os.path.join(config.PROJECT_ROOT, "checkpoints_rq2", mode_name)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    set_seed(config.SEED)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> [RQ2] Running Ablation: {mode_name}")

    # 2. 数据
    train_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, 'test'), batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. 模型
    model = get_model('Phys-RDLinear', config).to(device)

    # 4. 训练器 (带致盲功能)
    trainer = AblationTrainer(config, model, device, ablation_mode=mode_name)

    # 5. 训练
    for epoch in range(40):  # 20轮足够收敛
        metrics = trainer.train_epoch(train_dl)
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch + 1} | Loss: {metrics['loss']:.4f} | Train Acc: {metrics['acc']:.1f}%")

    # 6. 评估
    csv_path = os.path.join(config.CHECKPOINT_DIR, "eval_results.csv")
    acc = trainer.evaluate(test_dl, save_path=csv_path)
    print(f"    Eval Acc: {acc:.2f}%")

    return csv_path


def analyze_results(csv_paths):
    """分析三个实验的结果，生成包含所有故障类型的对比表格"""
    print("\n========================================================")
    print("   RQ2 Analysis: Resolution Sensitivity (Full Spectrum)")
    print("========================================================")

    stats = []

    for mode, path in csv_paths.items():
        if not os.path.exists(path):
            print(f"[Warn] Missing results for {mode}")
            continue

        df = pd.read_csv(path)

        # 1. 计算总体精度
        row_data = {"Configuration": mode, "Total Avg": df['Is_Correct'].mean() * 100}

        # 2. 遍历计算每一类故障的精度
        for cls_id, cls_name in FAULT_MAP.items():
            # 筛选该类样本
            subset = df[df['True_Label'] == cls_id]
            if len(subset) > 0:
                acc = subset['Is_Correct'].mean() * 100
            else:
                acc = 0.0

            # 使用简称作为列名，防止表格太宽
            # 例如: 'KA (Rotor Bar)' -> 'KA'
            short_name = cls_name.split(' ')[0]
            row_data[short_name] = acc

        stats.append(row_data)

    if not stats:
        print("No data collected.")
        return

    res_df = pd.DataFrame(stats)

    # 调整顺序
    order_map = {'no_micro': 0, 'no_macro': 1, 'full': 2}
    res_df['sort_key'] = res_df['Configuration'].map(order_map)
    res_df = res_df.sort_values('sort_key').drop('sort_key', axis=1)

    # 重命名索引，使其更具物理含义
    config_rename = {
        'no_micro': 'Macro-Only (High Freq)',
        'no_macro': 'Micro-Only (Low Freq)',
        'full': 'Multi-Res (Ours)'
    }
    res_df['Configuration'] = res_df['Configuration'].map(config_rename)

    # 设置 Configuration 为索引
    res_df = res_df.set_index('Configuration')

    # 格式化
    res_df = res_df.round(1)

    print("\n>>> Resolution Sensitivity Table:")
    print(res_df.to_string())

    res_df.to_csv("Table_RQ2_Full_Faults.csv")
    print("\n>>> Table saved to Table_RQ2_Full_Faults.csv")


if __name__ == "__main__":
    # 定义三个任务
    tasks = {
        'full': 'full',  # 双流
        'no_micro': 'no_micro',  # 只有全景 (51.2k)
        'no_macro': 'no_macro'  # 只有显微 (1k)
    }

    results = {}
    for task_name, mode in tasks.items():
        # 这里串行运行，如果你有多个GPU可以并行
        csv = run_ablation_task(mode, gpu_id=0)
        results[mode] = csv

    analyze_results(results)
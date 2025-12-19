import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed

# 故障映射表
FAULT_MAP = {
    0: 'HH', 1: 'RU', 2: 'RM', 3: 'SW',
    4: 'VU', 5: 'BR', 6: 'KA', 7: 'FB'
}


class AblationTrainer(Trainer):
    def __init__(self, config, model, device, ablation_mode='none'):
        super().__init__(config, model, device)
        self.ablation_mode = ablation_mode

    def _mask_input(self, micro, macro):
        """ 致盲核心：将不需要的分支置为 0 """
        if self.ablation_mode == 'no_micro':
            micro = torch.zeros_like(micro)
        elif self.ablation_mode == 'no_macro':
            macro = torch.zeros_like(macro)
        return micro, macro

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        # [Fix] 解包 6 个变量
        for micro, macro, acoustic, speed, y_cls, y_load in dataloader:
            micro, macro = micro.to(self.device), macro.to(self.device)
            acoustic, speed = acoustic.to(self.device), speed.to(self.device)
            y_cls, y_load = y_cls.to(self.device), y_load.to(self.device)

            # 执行致盲
            micro, macro = self._mask_input(micro, macro)

            self.optimizer.zero_grad()
            # 我们的模型现在统一接受声纹输入，如果没有声纹实验需求，acoustic 传入即可
            logits, pred_load = self.model(micro, macro, acoustic, speed)

            # 计算 Loss (支持 MTL)
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

        for micro, macro, acoustic, speed, y_cls, y_load in dataloader:
            micro, macro = micro.to(self.device), macro.to(self.device)
            acoustic, speed = acoustic.to(self.device), speed.to(self.device)

            micro, macro = self._mask_input(micro, macro)

            logits, _ = self.model(micro, macro, acoustic, speed)
            preds = torch.argmax(logits, dim=1)

            correct += (preds == y_cls).sum().item()
            total += y_cls.size(0)

            # 收集结果
            b_true = y_cls.cpu().numpy()
            b_pred = preds.cpu().numpy()
            for i in range(len(b_true)):
                results.append({
                    'True_Label': b_true[i],
                    'Is_Correct': 1 if b_true[i] == b_pred[i] else 0
                })

        if save_path:
            pd.DataFrame(results).to_csv(save_path, index=False)
        return 100 * correct / total


def run_rq2_task(mode_name, gpu_id=0):
    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'
    config.CHECKPOINT_DIR = os.path.join("checkpoints_rq2", mode_name)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    set_seed(config.SEED)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    print(f"\n>>> [RQ2] Running Ablation: {mode_name}")

    # 加载数据
    train_dl = DataLoader(Ch4DualStreamDataset(config, 'train'), batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, 'test'), batch_size=config.BATCH_SIZE, shuffle=False)

    # 加载模型
    model = get_model('Phys-RDLinear', config).to(device)

    # 训练
    trainer = AblationTrainer(config, model, device, ablation_mode=mode_name)
    for epoch in range(20):  # 20 Epochs 足够
        trainer.train_epoch(train_dl)

    # 评估
    csv_path = os.path.join(config.CHECKPOINT_DIR, "eval_results.csv")
    acc = trainer.evaluate(test_dl, save_path=csv_path)
    print(f"    Finished {mode_name}: Acc {acc:.2f}%")
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
    modes = ['full', 'no_micro', 'no_macro']

    results = {}
    for m in modes:
        results[m] = run_rq2_task(m)

    analyze_results(results)
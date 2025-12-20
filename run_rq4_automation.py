# run_rq4_advanced.py
import os
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from trainer import Trainer
from utils.tools import set_seed
from utils.rq4_kit import plot_rq4_modal_comparison

EXPERIMENTS = {
    'Vib_Only': {'desc': '仅振动', 'mask_vib': False, 'mask_audio': True, 'mask_curr': True},
    'Audio_Only': {'desc': '仅声纹', 'mask_vib': True, 'mask_audio': False, 'mask_curr': True},
    'Curr_Only': {'desc': '仅电流', 'mask_vib': True, 'mask_audio': True, 'mask_curr': False},  # [NEW]
    'Fusion': {'desc': '三模态融合', 'mask_vib': False, 'mask_audio': False, 'mask_curr': False}
}


class MaskedTrainer(Trainer):
    def __init__(self, config, model, device, mask_cfg):
        super().__init__(config, model, device)
        self.mask_cfg = mask_cfg

    def _apply_mask(self, mic, mac, ac, cur):
        if self.mask_cfg.get('mask_vib', False):
            mic = torch.zeros_like(mic)
            mac = torch.zeros_like(mac)
        if self.mask_cfg.get('mask_audio', False):
            ac = torch.zeros_like(ac)
        if self.mask_cfg.get('mask_curr', False):  # [NEW]
            cur = torch.zeros_like(cur)
        return mic, mac, ac, cur

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        # [V2 Change] Unpack 7
        for mic, mac, ac, cur, spd, ld, label in dataloader:
            mic, mac = mic.to(self.device), mac.to(self.device)
            ac, cur = ac.to(self.device), cur.to(self.device)
            spd, ld = spd.to(self.device), ld.to(self.device)
            label = label.to(self.device)

            # Masking
            mic, mac, ac, cur = self._apply_mask(mic, mac, ac, cur)

            self.optimizer.zero_grad()
            # Forward V2
            logits, _ = self.model(mic, mac, ac, cur, spd, ld)

            loss = self.criterion(logits, label)  # V2 trainer uses weighted CE
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return {'loss': total_loss / len(dataloader)}


def evaluate_rq4_local(model, dataloader, device, mask_cfg):
    """
    RQ4 专用评估逻辑
    关键：如果是 Audio_Only，必须禁用 PEG (物理能量门控)，否则所有样本会被判为健康
    """
    model.eval()
    all_preds, all_labels, all_loads = [], [], []

    # 智能 PEG 开关
    # 只有当振动信号存在 (mask_vib=False) 时，PEG 才有意义
    ENABLE_PEG = not mask_cfg['mask_vib']
    PEG_THRESHOLD = 0.05

    with torch.no_grad():
        for micro, macro, ac, spd, y_cls, y_load in dataloader:
            micro, macro, ac, spd = micro.to(device), macro.to(device), ac.to(device), spd.to(device)

            # Mask
            if mask_cfg['mask_vib']:
                micro = torch.zeros_like(micro)
                macro = torch.zeros_like(macro)
            if mask_cfg['mask_audio']:
                ac = torch.zeros_like(ac)

            # Inference
            # 强制 4 参数调用
            logits, _ = model(micro, macro, ac, spd)
            preds = torch.argmax(logits, dim=1)

            # PEG (仅在振动模式下启用)
            if ENABLE_PEG:
                input_rms = torch.sqrt(torch.mean(micro.squeeze(-1) ** 2, dim=1))
                is_noise = input_rms < PEG_THRESHOLD
                preds[is_noise] = 0

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y_cls.cpu().numpy())
            all_loads.append(y_load.cpu().numpy())

    # 汇总指标
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    loads_real = (np.concatenate(all_loads).flatten() * 400).round().astype(int)

    metrics = {}
    metrics['Avg_Acc'] = accuracy_score(y_true, y_pred) * 100
    metrics['Avg_F1'] = f1_score(y_true, y_pred, average='macro') * 100

    # 分负载计算
    for ld in [0, 400]:
        mask = (loads_real == ld)
        if np.sum(mask) > 0:
            acc = accuracy_score(y_true[mask], y_pred[mask]) * 100
            metrics[f'Acc_{ld}kg'] = acc
        else:
            metrics[f'Acc_{ld}kg'] = 0.0

    return metrics


def main():
    print("=========================================================")
    print("   RQ4 Automation: Multi-Modal Ablation Study")
    print("=========================================================")

    rq4_root = os.path.join("checkpoints_ch4", "rq4")
    os.makedirs(rq4_root, exist_ok=True)

    config = Ch4Config()
    config.MODEL_NAME = 'Phys-RDLinear'  # RQ4 固定模型
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    train_dl = DataLoader(Ch4DualStreamDataset(config, mode='train'),
                          batch_size=config.BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(Ch4DualStreamDataset(config, mode='test'),
                         batch_size=config.BATCH_SIZE, shuffle=False)

    results = []

    for cfg_name, mask_settings in EXPERIMENTS.items():
        print(f"\n>>> [Task] Running: {cfg_name} ({mask_settings['desc']})")

        save_dir = os.path.join(rq4_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)

        # 1. 初始化
        model = get_model('Phys-RDLinear', config).to(device)

        # 2. 训练
        trainer = MaskedTrainer(config, model, device, mask_settings)
        ckpt_path = os.path.join(save_dir, "model.pth")

        if os.path.exists(ckpt_path):
            print("    [Info] Loading checkpoint...")
            # 增加 weights_only=False 避免警告 (或按需处理)
            model.load_state_dict(torch.load(ckpt_path))
        else:
            print("    [Info] Training...")
            epochs = 30
            for epoch in range(epochs):
                trainer.train_epoch(train_dl)
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估
        print("    [Eval] Evaluating...")
        metrics = evaluate_rq4_local(model, test_dl, device, mask_settings)

        row = {
            'Config': cfg_name,
            'Description': mask_settings['desc'],
            'Avg_Acc': metrics['Avg_Acc'],
            'Acc_0kg': metrics['Acc_0kg'],
            'Acc_400kg': metrics['Acc_400kg']
        }
        results.append(row)
        print(f"    -> Avg Acc: {row['Avg_Acc']:.2f}% (0kg: {row['Acc_0kg']:.1f}%)")

    # 4. 汇总与绘图
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(rq4_root, "RQ4_Modal_Analysis.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> 汇总表已保存: {csv_path}")

        plot_rq4_modal_comparison(csv_path, rq4_root)

    print("\n=========================================================")
    print("   RQ4 Completed! See checkpoints/rq4/")
    print("=========================================================")


if __name__ == "__main__":
    main()
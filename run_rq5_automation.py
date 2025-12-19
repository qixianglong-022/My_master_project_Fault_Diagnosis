import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# 导入项目模块
from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.factory import get_model
from utils.tools import set_seed


# === 1. 噪声注入工具 ===
class NoiseInjector:
    @staticmethod
    def add_noise(tensor, snr_db):
        if snr_db is None: return tensor
        sig_power = torch.mean(tensor ** 2, dim=(1, 2), keepdim=True)
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(tensor) * noise_std
        return tensor + noise


# === 2. 主实验逻辑 ===
def run_rq5_experiments():
    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n========================================================")
    print("   RQ5: Robustness Test (Formerly RQ4)")
    print("========================================================")

    models_to_test = ['FD-CNN', 'TiDE', 'Phys-RDLinear']
    snr_levels = [None, 10, 5, 0, -5]

    test_ds = Ch4DualStreamDataset(config, 'test')
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    noise_results = []

    for model_name in models_to_test:
        print(f"\n>>> Testing Model: {model_name}")
        try:
            model = get_model(model_name, config).to(device)
            # 优先加载 RQ1/RQ3 的权重
            ckpt_path = os.path.join("checkpoints_ch4", model_name, "model.pth")
            if not os.path.exists(ckpt_path) and model_name == 'Phys-RDLinear':
                ckpt_path = os.path.join("checkpoints_rq3", "Phys-RDLinear", "model.pth")

            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"    Loaded weights from {ckpt_path}")
            else:
                print(f"    [Warn] No weights found, using random init!")
        except Exception as e:
            print(f"    [Error] Failed to load: {e}")
            continue

        model.eval()
        row_res = {'Model': model_name}

        for snr in snr_levels:
            correct, total = 0, 0
            with torch.no_grad():
                for micro_x, macro_x, _, speed, y_cls, y_load in test_dl:  # 注意这里解包占位符 _ (acoustic)
                    mask = (y_load < 0.25).flatten()
                    if not mask.any(): continue

                    micro_x = micro_x[mask].to(device)
                    macro_x = macro_x[mask].to(device)
                    speed = speed[mask].to(device)
                    y_cls = y_cls[mask].to(device)

                    micro_noisy = NoiseInjector.add_noise(micro_x, snr)
                    macro_noisy = NoiseInjector.add_noise(macro_x, snr)

                    if model_name == 'Phys-RDLinear':
                        # RQ5 测试的是旧模型，不带 acoustic
                        logits, _ = model(micro_noisy, macro_noisy, speed)
                    else:
                        full_x = torch.cat([micro_noisy, macro_noisy], dim=1)
                        logits, _ = model(full_x, speed)

                    preds = torch.argmax(logits, dim=1)
                    correct += (preds == y_cls).sum().item()
                    total += y_cls.size(0)

            acc = 100 * correct / total if total > 0 else 0
            col_name = f"Clean" if snr is None else f"{snr}dB"
            row_res[col_name] = acc
            print(f"    SNR={col_name:>5} | Acc: {acc:.2f}%")

        noise_results.append(row_res)

    if noise_results:
        df_noise = pd.DataFrame(noise_results)
        df_noise = df_noise.round(2)
        print("\n>>> Table 5.1: Noise Robustness (Target: 0kg)")
        print(df_noise.to_string(index=False))
        df_noise.to_csv("Table_RQ5_Noise_Robustness.csv", index=False)


if __name__ == "__main__":
    run_rq5_experiments()
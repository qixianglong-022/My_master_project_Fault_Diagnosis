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
    """
    在线噪声注入器
    """

    @staticmethod
    def add_noise(tensor, snr_db):
        """
        向张量注入指定信噪比(SNR)的高斯白噪声
        tensor: [Batch, Channel, Length]
        """
        if snr_db is None:
            return tensor

        # 1. 计算信号功率 (P_signal)
        # 沿最后两个维度计算均方值
        sig_power = torch.mean(tensor ** 2, dim=(1, 2), keepdim=True)

        # 2. 计算噪声功率 (P_noise)
        # SNR_db = 10 * log10(P_s / P_n)  => P_n = P_s / 10^(SNR/10)
        noise_power = sig_power / (10 ** (snr_db / 10))

        # 3. 生成噪声
        # sigma = sqrt(P_noise)
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(tensor) * noise_std

        return tensor + noise


# === 2. 主实验逻辑 ===
def run_rq4_experiments():
    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n========================================================")
    print("   RQ4: Noise Robustness Test (Target: 0kg Light Load)")
    print("========================================================")

    # 定义要对比的模型
    # 确保这些模型名称与 checkpoints_ch4 下的文件夹对应
    models_to_test = ['FD-CNN', 'TiDE', 'Phys-RDLinear']

    # 定义噪声等级: Clean, 10dB, 5dB, 0dB, -5dB (噪音比信号还大)
    snr_levels = [None, 10, 5, 0, -5]

    # 准备测试数据
    test_ds = Ch4DualStreamDataset(config, 'test')
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # 存储结果
    noise_results = []

    for model_name in models_to_test:
        print(f"\n>>> Testing Model: {model_name}")

        # 1. 加载模型
        try:
            model = get_model(model_name, config).to(device)
            # 路径兼容：优先找 RQ1 的 checkpoints_ch4，找不到找 RQ3
            ckpt_path = os.path.join("checkpoints_ch4", model_name, "model.pth")
            if not os.path.exists(ckpt_path) and model_name == 'Phys-RDLinear':
                ckpt_path = os.path.join("checkpoints_rq3", "Phys-RDLinear", "model.pth")

            if os.path.exists(ckpt_path):
                # weights_only=False 警告可忽略
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
                print(f"    Loaded weights from {ckpt_path}")
            else:
                print(f"    [Warn] No weights found for {model_name}, using random init!")
        except Exception as e:
            print(f"    [Error] Failed to load {model_name}: {e}")
            continue

        model.eval()

        # === Experiment: Noise Robustness (Target: 0kg Light Load) ===
        row_res = {'Model': model_name}

        for snr in snr_levels:
            correct, total = 0, 0

            with torch.no_grad():
                for micro_x, macro_x, speed, y_cls, y_load in test_dl:
                    # 筛选 0kg 数据 (归一化 Load < 0.25 即视为轻载)
                    # y_load shape: [B, 1]
                    mask = (y_load < 0.25).flatten()
                    if not mask.any(): continue

                    # 应用筛选
                    micro_x = micro_x[mask].to(device)
                    macro_x = macro_x[mask].to(device)
                    speed = speed[mask].to(device)
                    y_cls = y_cls[mask].to(device)

                    # 注入噪声
                    micro_noisy = NoiseInjector.add_noise(micro_x, snr)
                    macro_noisy = NoiseInjector.add_noise(macro_x, snr)

                    # 前向传播
                    if model_name == 'Phys-RDLinear':
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

    # === 2. 生成报表 ===
    if noise_results:
        df_noise = pd.DataFrame(noise_results)
        df_noise = df_noise.round(2)
        print("\n>>> Table 4.6: Noise Robustness (Target: 0kg)")
        print(df_noise.to_string(index=False))
        df_noise.to_csv("Table_RQ4_Noise_Robustness.csv", index=False)
        print("\n>>> Table saved to Table_RQ4_Noise_Robustness.csv")

    print("\n>>> RQ4 Experiment Completed.")


if __name__ == "__main__":
    run_rq4_experiments()
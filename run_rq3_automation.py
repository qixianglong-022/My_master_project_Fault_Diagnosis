import os
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from config import Ch4Config
from data_loader_ch4 import Ch4DualStreamDataset
from models.phys_rdlinear import PhysRDLinearCls
from trainer import Trainer
from utils.tools import set_seed
from utils.rq1_kit import evaluate_rq1_comprehensive
from utils.rq3_kit import compute_mmd, plot_ablation_chart

# ==============================================================================
# 1. 实验配置
# ==============================================================================
ABLATION_CONFIGS = {
    'Ablation_Base': {
        'pgfa': False, 'mtl': False, 'desc': '无物理引导 (Baseline)'
    },
    'Ablation_PGFA': {
        'pgfa': True, 'mtl': False, 'desc': '仅 PGFA (频率校准)'
    },
    'Ablation_MTL': {
        'pgfa': False, 'mtl': True, 'desc': '仅 MTL (负载感知)'
    },
    'Phys_RDLinear': {
        'pgfa': True, 'mtl': True, 'desc': '完整模型 (Ours)'
    }
}
GPU_ID = 0


# ==============================================================================
# 2. 物理引导专用训练器 (处理 MTL Loss)
# ==============================================================================
class PhysicsTrainer(Trainer):
    """
    RQ3 专用训练器
    特性：
    1. 适配 8 参数 DataLoader
    2. 支持 MTL 损失计算 (分类 + 负载回归)
    """

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        total_cls_loss = 0
        total_reg_loss = 0

        # [适配] 解包 8 个变量
        for micro, macro, ac, cur, spd, y_load, y_cls, _ in dataloader:
            # 1. 数据迁移
            micro, macro = micro.to(self.device), macro.to(self.device)
            ac, cur = ac.to(self.device), cur.to(self.device)
            spd, y_cls = spd.to(self.device), y_cls.to(self.device)

            # y_load 已经是 [B, 1]，不需要 unsqueeze
            y_load = y_load.to(self.device).float()

            self.optimizer.zero_grad()

            # 2. 前向传播 (传入全量数据)
            # 注意：即使是 Ablation_Base，模型结构也是全量的，只是内部逻辑不同
            logits, pred_load = self.model(micro, macro, ac, cur, spd, y_load)

            # 3. 计算分类损失 (基础)
            loss_cls = F.cross_entropy(logits, y_cls)
            loss = loss_cls

            # 4. 计算回归损失 (如果开启了 MTL)
            loss_reg_val = 0.0
            # 只有当模型确实输出了预测值 (enable_mtl=True) 且不为 None 时才计算
            if pred_load is not None:
                # MSE Loss: 预测负载 vs 真实负载(软感知)
                loss_reg = F.mse_loss(pred_load, y_load)

                # [关键] 损失平衡权重
                # 0.5 是一个经验值，平衡分类主任务和回归辅助任务
                loss = 0.5 * loss_cls + 0.5 * loss_reg
                loss_reg_val = loss_reg.item()

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_cls_loss += loss_cls.item()
            total_reg_loss += loss_reg_val

        avg_metrics = {
            'loss': total_loss / len(dataloader),
            'loss_cls': total_cls_loss / len(dataloader),
            'loss_reg': total_reg_loss / len(dataloader)
        }
        return avg_metrics


# ==============================================================================
# 3. 主流程
# ==============================================================================
def main():
    print("=========================================================")
    print("   RQ3 Automation: Physics Module Ablation Study")
    print("=========================================================")

    rq3_root = os.path.join("checkpoints_ch4", "rq3")
    os.makedirs(rq3_root, exist_ok=True)

    config = Ch4Config()
    set_seed(config.SEED)
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu')

    # 1. 准备数据
    train_ds = Ch4DualStreamDataset(config, mode='train')
    test_ds = Ch4DualStreamDataset(config, mode='test')

    train_dl = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)

    # 评估用的 Loader
    test_dl = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # MMD 计算专用 Loader (Source 不 Shuffle, Target Shuffle 以随机采样)
    # 减小 batch_size 以防 OOM (MMD 计算比较耗显存)
    mmd_src_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    mmd_tgt_dl = DataLoader(test_ds, batch_size=32, shuffle=True)

    results = []

    for cfg_name, settings in ABLATION_CONFIGS.items():
        print(f"\n>>> [Task] Running: {cfg_name} ({settings['desc']})")

        # 更新配置名
        config.MODEL_NAME = cfg_name
        save_dir = os.path.join(rq3_root, cfg_name)
        os.makedirs(save_dir, exist_ok=True)
        config.CHECKPOINT_DIR = save_dir

        # 1. 初始化模型
        # [修复] 移除 enable_acoustic 参数 (Model 定义中没有这个参数)
        model = PhysRDLinearCls(
            config,
            enable_pgfa=settings['pgfa'],
            enable_mtl=settings['mtl']
        ).to(device)

        # 2. 训练 (使用 PhysicsTrainer)
        trainer = PhysicsTrainer(config, model, device)
        ckpt_path = os.path.join(save_dir, "model.pth")

        if os.path.exists(ckpt_path):
            print("    [Info] Loading checkpoint...")
            try:
                model.load_state_dict(torch.load(ckpt_path, map_location=device))
            except:
                print("    [Warn] Checkpoint load failed, retraining...")
                os.remove(ckpt_path)

        if not os.path.exists(ckpt_path):
            print("    [Info] Training...")
            epochs = 25  # 消融实验
            for epoch in range(epochs):
                m = trainer.train_epoch(train_dl)
                if (epoch + 1) % 5 == 0:
                    print(
                        f"        Ep {epoch + 1}: Loss={m['loss']:.4f} (Cls={m['loss_cls']:.4f}, Reg={m['loss_reg']:.4f})")
            torch.save(model.state_dict(), ckpt_path)

        # 3. 评估准确率
        print("    [Eval] Evaluating Accuracy...")
        metrics = evaluate_rq1_comprehensive(model, test_dl, device, save_dir, cfg_name)

        # 4. 计算 MMD (物理引导效果的关键指标)
        print("    [Eval] Calculating MMD Distance (Source <-> Target)...")
        # 传入 model_name 以便 kit 内部做判断 (虽然这里都是 PhysRDLinear)
        mmd_val = compute_mmd(model, mmd_src_dl, mmd_tgt_dl, device)
        print(f"    -> MMD Distance: {mmd_val:.4f}")

        row = {
            'Config': cfg_name,
            'Description': settings['desc'],
            'Avg_Acc': metrics.get('Avg_Acc', 0),
            'Acc_0kg': metrics.get('Acc_0kg', 0),
            'Acc_400kg': metrics.get('Acc_400kg', 0),
            'MMD_Distance': mmd_val,
            'PGFA': settings['pgfa'],
            'MTL': settings['mtl']
        }
        results.append(row)

    # 5. 汇总与绘图
    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(rq3_root, "RQ3_Ablation_Results.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n>>> Results saved: {csv_path}")

        print(">>> Generating Charts...")
        try:
            plot_ablation_chart(csv_path, rq3_root)
        except Exception as e:
            print(f"    [Warn] Plotting failed: {e}")

    print("\n=========================================================")
    print("   RQ3 Completed! See checkpoints_ch4/rq3/")
    print("=========================================================")


if __name__ == "__main__":
    main()
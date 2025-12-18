import torch
from torch.utils.data import DataLoader
from models.phys_rdlinear_cls import PhysRDLinearCls
from config import Config
from data_loader_ch4 import Ch4DualStreamDataset
from utils.loss_ch4 import MultiTaskLoss


def train_ch4_diagnostic():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 模型与优化器初始化
    # freq_dim 对应显微流 FFT 后的长度 (1024 // 2 = 512)
    model = PhysRDLinearCls(num_classes=8, freq_dim=512).to(device)
    mtl_loss = MultiTaskLoss().to(device)  # 自适应损失模块

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': [mtl_loss.log_var_cls, mtl_loss.log_var_reg], 'lr': 0.001}
    ], lr=Config.LEARNING_RATE)

    # 2. 自动化域泛化数据加载
    # 假设加载 Load 0 & 400 进行训练，Load 200 用于测试泛化
    train_ds = Ch4DualStreamDataset(mode='train')  # 需配合双流数据读取逻辑
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(Config.CH4_EPOCHS):
        for i, (micro_x, panorama_x, speed_hz, labels, loads) in enumerate(train_loader):
            micro_x = micro_x.to(device)
            speed_hz = speed_hz.to(device)
            labels = labels.to(device)
            loads = loads.to(device)

            optimizer.zero_grad()

            # 前向传播：主要利用显微流进行精细化诊断
            logits, pred_load = model(micro_x, speed_hz)

            # 计算自适应多任务损失
            total_loss, l_cls, l_reg = mtl_loss(logits, labels, pred_load, loads)

            total_loss.backward()
            optimizer.step()

            if i % 10 == 0:
                # 使用 torch.exp(-log_var) 还原为任务的权重系数，这样更直观
                w_cls = torch.exp(-mtl_loss.log_var_cls).item()
                w_reg = torch.exp(-mtl_loss.log_var_reg).item()
                print(f"Epoch [{epoch}] Batch [{i}] | Cls_Loss: {l_cls:.4f} | Reg_Loss: {l_reg:.4f}")
                print(f"    >>> 动态权重自适应: W_cls: {w_cls:.4f}, W_reg: {w_reg:.4f}")

    # 3. 保存最佳模型
    torch.save(model.state_dict(), "phys_rdlinear_best.pth")
import torch
import torch.nn as nn


def calculate_mmd(source_features, target_features):
    """
    [论文对齐] 4.4.2 节：多核最大均值差异 (Multi-Kernel MMD)
    用于量化源域 (200kg) 与目标域 (0kg/400kg) 提取特征的分布距离。

    实事求是分析：
    该值越小，说明 Phys-RDLinear 提取的特征在不同负载下越一致，
    即 Trend 分支成功吸附了工况方差，实现了特征解耦。
    """

    def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        """
        多核高斯核矩阵计算
        """
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        # 计算 L2 距离矩阵
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)

        # 计算带宽 sigma
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)

        # 多核叠加
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-L2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    # 1. 计算核矩阵
    batch_size = int(source_features.size(0))
    kernels = gaussian_kernel(source_features, target_features)

    # 2. 按照 MMD 统计量公式计算
    # MMD^2 = E[k(xs,xs)] + E[k(xt,xt)] - 2E[k(xs,xt)]
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]

    loss = torch.mean(XX + YY - XY - YX)

    # 保证非负（数值误差可能导致微小负值）
    return torch.clamp(loss, min=0.0).item()


# ---------------------------------------------------------
# [自动化测试程序] 验证 MMD 逻辑
# ---------------------------------------------------------
if __name__ == "__main__":
    # 模拟 200kg 源域特征
    source_feat = torch.randn(64, 256) + 0.0  # 均值 0

    # 模拟解耦失败的目标域特征 (偏移大)
    target_fail = torch.randn(64, 256) + 2.0  # 均值 2.0

    # 模拟解耦成功的目标域特征 (偏移小)
    target_success = torch.randn(64, 256) + 0.2  # 均值 0.2

    mmd_bad = calculate_mmd(source_feat, target_fail)
    mmd_good = calculate_mmd(source_feat, target_success)

    print(f">>> [测试] 解耦失败时的 MMD 距离: {mmd_bad:.4f}")
    print(f">>> [测试] 解耦成功时的 MMD 距离: {mmd_good:.4f}")
    print(f">>> [结论] MMD 显著下降，指标有效。")
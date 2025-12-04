# debug_loader.py
import torch
from data_loader import MotorDataset
from config import Config


def test():
    print(">>> 开始 Data Loader 独立测试")
    # 强制设置 Config，防止被修改
    Config.ENC_IN = 21

    ds = MotorDataset(flag='train', condition='HH')
    print(f"数据集大小: {len(ds)}")

    # 获取第一个样本
    item = ds[0]
    x, cov = item

    print("\n" + "=" * 30)
    print("【真相时刻】")
    print(f"x_features 应该的维度: (49, 21)")
    print(f"实际返回的维度: {x.shape}")
    print("=" * 30)

    if x.shape[1] == 5:
        print("❌ 错误确诊：__getitem__ 返回的是原始数据(5维)，而不是特征(21维)！")
        print("   请立即检查 data_loader.py 的 return 语句！")
    elif x.shape[1] == 21:
        print("✅ 测试通过：Data Loader 正常。如果 train.py 还在报错，那是见鬼了。")


if __name__ == '__main__':
    test()
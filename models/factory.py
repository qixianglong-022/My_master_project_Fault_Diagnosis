import torch
from config import Ch4Config
from models.phys_rdlinear import PhysRDLinearCls
from models.baselines_ch4 import FD_CNN, TiDE_Cls, ResNet18_2D, Vanilla_RDLinear_Cls


# models/factory.py

def get_model(model_name: str, config: Ch4Config):
    """
    模型工厂：根据名称实例化对应模型
    """
    print(f"Factory initializing: {model_name}")

    # [删除或注释掉这行] 不要强制翻倍，除非你真的拼接了 micro 和 macro
    # FULL_DIM = config.FREQ_DIM * 2

    # [新增] 明确定义基线使用的维度 = Micro 维度
    BASE_DIM = config.FREQ_DIM

    # 1. 我们的主角 (Ours) - 内部会自己处理 Micro/Macro，不受这里影响
    if model_name == 'Phys-RDLinear':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=False)

    # 2. 消融实验配置
    elif model_name == 'Ablation-Base':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=False)
    elif model_name == 'Ablation-PGFA':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=False)
    elif model_name == 'Ablation-MTL':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=True)

    # 3. 基线模型 (Baselines) - 统统改为使用 BASE_DIM (512)
    elif model_name == 'FD-CNN':
        return FD_CNN(num_classes=config.NUM_CLASSES, freq_dim=BASE_DIM)

    elif model_name == 'TiDE':
        return TiDE_Cls(num_classes=config.NUM_CLASSES, freq_dim=BASE_DIM)

    elif model_name == 'ResNet-18':
        return ResNet18_2D(num_classes=config.NUM_CLASSES, input_len=BASE_DIM)

    elif model_name == 'Vanilla RDLinear':
        return Vanilla_RDLinear_Cls(num_classes=config.NUM_CLASSES, freq_dim=BASE_DIM)

    else:
        raise ValueError(f"Unknown Model Name: {model_name}")
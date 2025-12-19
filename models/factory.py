import torch
from config import Ch4Config
from models.phys_rdlinear import PhysRDLinearCls
from models.baselines_ch4 import FD_CNN, TiDE_Cls, ResNet18_2D, Vanilla_RDLinear_Cls


def get_model(model_name: str, config: Ch4Config):
    """
    模型工厂：根据名称实例化对应模型
    """
    print(f"Factory initializing: {model_name}")

    # Micro (512) + Macro (512) = 1024
    FULL_DIM = config.FREQ_DIM * 2

    # 1. 我们的主角 (Ours)
    if model_name == 'Phys-RDLinear':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=True)

    # 2. 消融实验配置
    elif model_name == 'Ablation-Base':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=False)
    elif model_name == 'Ablation-PGFA':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=False)
    elif model_name == 'Ablation-MTL':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=True)

    # 3. 基线模型 (Baselines)
    elif model_name == 'FD-CNN':
        return FD_CNN(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    elif model_name == 'TiDE':
        return TiDE_Cls(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    elif model_name == 'ResNet-18':
        # [修改点 2] 直接使用顶部导入的类，无需并在函数内 import
        return ResNet18_2D(num_classes=config.NUM_CLASSES, input_len=FULL_DIM)

    elif model_name == 'Vanilla RDLinear':
        return Vanilla_RDLinear_Cls(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    else:
        raise ValueError(f"Unknown Model Name: {model_name}")
import torch
from config import Ch4Config
from models.phys_rdlinear import PhysRDLinearCls
from models.baselines_ch4 import FD_CNN, TiDE_Cls, ResNet18_Thin, Vanilla_RDLinear_Cls


def get_model(model_name: str, config: Ch4Config):
    """
    模型工厂：根据名称实例化对应模型
    """
    print(f"Factory initializing: {model_name}")

    # === 关键修改：计算双流拼接后的总维度 ===
    # Micro (512) + Macro (512) = 1024
    FULL_DIM = config.FREQ_DIM * 2

    # 1. 我们的主角 (Ours) - 内部自己处理双流，不需要传 FULL_DIM
    if model_name == 'Phys-RDLinear':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=True)

    # 2. Config A: Base (无 PGFA, 无 MTL)
    elif model_name == 'Ablation-Base':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=False)

    # 3. Config B: Base + PGFA
    elif model_name == 'Ablation-PGFA':
        return PhysRDLinearCls(config, enable_pgfa=True, enable_mtl=False)

    # 4. Config C: Base + MTL
    elif model_name == 'Ablation-MTL':
        return PhysRDLinearCls(config, enable_pgfa=False, enable_mtl=True)

    # 2. 基线模型 (Baselines) - 必须传入 FULL_DIM
    elif model_name == 'FD-CNN':
        return FD_CNN(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    elif model_name == 'TiDE':
        return TiDE_Cls(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    elif model_name == 'ResNet-18':
        # ResNet 需要知道总长度来 reshape
        return ResNet18_Thin(num_classes=config.NUM_CLASSES, input_len=FULL_DIM)

    elif model_name == 'Vanilla RDLinear':
        return Vanilla_RDLinear_Cls(num_classes=config.NUM_CLASSES, freq_dim=FULL_DIM)

    else:
        raise ValueError(f"Unknown Model Name: {model_name}")
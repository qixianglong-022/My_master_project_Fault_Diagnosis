#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : export_to_edge.py
@Desc    : 自动化模型导出脚本 - 支持第三章和第四章模型导出为ONNX格式
@Author  : Auto-generated
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 导入配置和模型
from config import Config, Ch4Config
from models.factory import get_model
from models.phys_rdlinear import PhysRDLinearCls
from models.baselines_ch4 import FD_CNN, TiDE_Cls, ResNet18_2D, Vanilla_RDLinear_Cls
from models.rdlinear import RDLinear
from models.baselines import LSTMAE, VanillaDLinear, TiDE, TransformerBaseline



class ModelExporter:
    """
    优雅的模型导出器 - 自动检测模型类型并生成ONNX
    """

    def __init__(self, checkpoint_path: str, output_dir: str, model_name: Optional[str] = None):
        """
        Args:
            checkpoint_path: 模型权重路径 (.pth)
            output_dir: 导出文件保存目录
            model_name: 模型名称（可选，自动检测）
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 自动检测模型类型
        self.model_name = model_name or self._detect_model_name()
        self.is_ch4 = self._is_chapter4_model()

        print(f"\n{'=' * 60}")
        print(f">>> Model Exporter Initialized")
        print(f"    Model: {self.model_name}")
        print(f"    Chapter: {'4' if self.is_ch4 else '3'}")
        print(f"    Checkpoint: {self.checkpoint_path}")
        print(f"    Output Dir: {self.output_dir}")
        print(f"{'=' * 60}\n")

    def _detect_model_name(self) -> str:
        """从checkpoint路径自动检测模型名称"""
        path_str = str(self.checkpoint_path)

        # 尝试从路径中提取模型名
        if 'Phys-RDLinear' in path_str or 'PhysRDLinear' in path_str:
            return 'Phys-RDLinear'
        elif 'RDLinear' in path_str:
            return 'RDLinear'
        elif 'LSTMAE' in path_str or 'LSTM_AE' in path_str:
            return 'LSTM_AE'
        elif 'DLinear' in path_str:
            return 'DLinear'
        elif 'TiDE' in path_str:
            return 'TiDE'
        elif 'ResNet' in path_str:
            return 'ResNet-18'
        elif 'FD-CNN' in path_str or 'FD_CNN' in path_str:
            return 'FD-CNN'
        elif 'Vanilla' in path_str:
            return 'Vanilla RDLinear'
        else:
            # 默认尝试第四章模型
            return 'Phys-RDLinear'

    def _is_chapter4_model(self) -> bool:
        """判断是否为第四章模型"""
        ch4_indicators = [
            'ch4', 'chapter4', 'checkpoints_ch4',
            'Phys-RDLinear', 'FD-CNN', 'TiDE', 'ResNet-18', 'Vanilla RDLinear'
        ]
        path_str = str(self.checkpoint_path).lower()
        return any(ind.lower() in path_str for ind in ch4_indicators)

    def load_model(self) -> nn.Module:
        """加载模型实例"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.is_ch4:
            config = Ch4Config()
            config.MODEL_NAME = self.model_name
            model = get_model(self.model_name, config)
        else:
            # 第三章模型加载逻辑
            if self.model_name == 'RDLinear':
                model = RDLinear()
            elif self.model_name == 'LSTMAE':
                model = LSTMAE(Config)
            elif self.model_name == 'DLinear':
                model = VanillaDLinear(Config)
            elif self.model_name == 'TiDE':
                model = TiDE(Config)
            elif self.model_name == 'Transformer':
                model = TransformerBaseline(Config)
            else:
                raise ValueError(f"Unknown Chapter 3 model: {self.model_name}")

        # 加载权重
        print(f">>> Loading checkpoint from: {self.checkpoint_path}")
        try:
            state_dict = torch.load(self.checkpoint_path, map_location=device)
            # 处理可能的键名不匹配（例如 'module.' 前缀）
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            model.to(device)
            print(f"    ✓ Model loaded successfully")
        except Exception as e:
            print(f"    ✗ Failed to load checkpoint: {e}")
            print(f"    Attempting to load with strict=False...")
            try:
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                model.to(device)
                print(f"    ✓ Model loaded with strict=False (some keys may be missing)")
            except Exception as e2:
                print(f"    ✗ Failed to load checkpoint even with strict=False: {e2}")
                raise

        return model

    def create_dummy_inputs_ch4(self) -> Tuple[torch.Tensor, ...]:
        """创建第四章模型的虚拟输入"""
        config = Ch4Config()
        batch_size = 1

        # 根据模型类型创建不同的输入
        if self.model_name == 'Phys-RDLinear':
            micro = torch.randn(batch_size, config.FREQ_DIM, 1)
            macro = torch.randn(batch_size, config.FREQ_DIM, 1)
            acoustic = torch.randn(batch_size, config.AUDIO_DIM)
            current_spec = torch.randn(batch_size, config.CURRENT_DIM)
            speed = torch.randn(batch_size, 1) * 50.0  # 0-50 Hz
            load_proxy = torch.randn(batch_size, 1) * 0.5 + 1.0  # 归一化负载
            return (micro, macro, acoustic, current_spec, speed, load_proxy)

        elif self.model_name in ['FD-CNN', 'Vanilla RDLinear']:
            # 这些模型只需要 micro 频谱
            micro = torch.randn(batch_size, config.FREQ_DIM, 1)
            speed = torch.randn(batch_size, 1) * 50.0
            return (micro, speed)

        elif self.model_name == 'TiDE':
            # TiDE 需要 flattened 输入
            micro = torch.randn(batch_size, config.FREQ_DIM, 1)
            speed = torch.randn(batch_size, 1) * 50.0
            return (micro, speed)

        elif self.model_name == 'ResNet-18':
            # ResNet 只需要频谱，但forward可能接受speed参数（可选）
            micro = torch.randn(batch_size, config.FREQ_DIM, 1)
            # 根据实际forward签名，ResNet可能不需要speed，但传入也无害
            return (micro,)

        else:
            raise ValueError(f"Unknown Chapter 4 model: {self.model_name}")

    def create_dummy_inputs_ch3(self) -> Tuple[torch.Tensor, ...]:
        """创建第三章模型的虚拟输入"""
        batch_size = 1
        seq_len = Config.WINDOW_SIZE
        feat_dim = Config.ENC_IN

        # 特征序列
        x = torch.randn(batch_size, seq_len, feat_dim)
        # 协变量 (速度相关)
        cov = torch.randn(batch_size, 3)

        return (x, cov)

    def export_onnx(self, model: nn.Module, dummy_inputs: Tuple, output_path: Path):
        """导出ONNX模型"""
        device = next(model.parameters()).device

        # 确保输入在正确的设备上
        dummy_inputs = tuple(inp.to(device) for inp in dummy_inputs)

        print(f">>> Exporting ONNX model...")
        print(f"    Input shapes: {[inp.shape for inp in dummy_inputs]}")

        try:
            # 获取输入名称
            if self.is_ch4:
                if self.model_name == 'Phys-RDLinear':
                    input_names = ['micro', 'macro', 'acoustic', 'current_spec', 'speed', 'load_proxy']
                elif self.model_name in ['FD-CNN', 'Vanilla RDLinear', 'TiDE']:
                    input_names = ['micro', 'speed']
                elif self.model_name == 'ResNet-18':
                    input_names = ['micro']
                else:
                    input_names = [f'input_{i}' for i in range(len(dummy_inputs))]
            else:
                input_names = ['signal', 'cov']

            output_names = ['logits'] if self.is_ch4 else ['reconstruction']

            # 对于第四章模型，需要包装forward以只返回logits
            if self.is_ch4:
                original_forward = model.forward

                def wrapped_forward(*args, **kwargs):
                    result = original_forward(*args, **kwargs)
                    # 如果返回tuple，只取第一个元素（logits）
                    if isinstance(result, tuple):
                        return result[0]
                    return result

                model.forward = wrapped_forward

            # 导出ONNX
            torch.onnx.export(
                model,
                dummy_inputs,
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=self._get_dynamic_axes(),
                opset_version=14,
                do_constant_folding=True,
                export_params=True,
                verbose=False
            )

            # 恢复原始forward（如果被包装了）
            if self.is_ch4:
                model.forward = original_forward

            print(f"    ✓ ONNX model saved to: {output_path}")

            # 验证ONNX模型
            try:
                import onnx
                onnx_model = onnx.load(str(output_path))
                onnx.checker.check_model(onnx_model)
                print(f"    ✓ ONNX model validation passed")
            except ImportError:
                print(f"    ⚠ onnx package not available, skipping validation")
            except Exception as e:
                print(f"    ⚠ ONNX validation warning: {e}")

        except Exception as e:
            print(f"    ✗ ONNX export failed: {e}")
            raise

    def _get_dynamic_axes(self) -> Dict[str, Any]:
        """获取动态轴配置（支持batch size变化）"""
        if self.is_ch4:
            if self.model_name == 'Phys-RDLinear':
                return {
                    'micro': {0: 'batch_size'},
                    'macro': {0: 'batch_size'},
                    'acoustic': {0: 'batch_size'},
                    'current_spec': {0: 'batch_size'},
                    'speed': {0: 'batch_size'},
                    'load_proxy': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
            else:
                return {
                    'micro': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                }
        else:
            return {
                'signal': {0: 'batch_size'},
                'cov': {0: 'batch_size'},
                'reconstruction': {0: 'batch_size'}
            }

    def save_scaler(self, scaler_path: Optional[str] = None):
        """保存标准化参数"""
        if scaler_path and os.path.exists(scaler_path):
            target_path = self.output_dir / "scaler_params.pkl"
            import shutil
            shutil.copy(scaler_path, target_path)
            print(f"    ✓ Scaler copied to: {target_path}")
        else:
            # 尝试自动查找scaler
            possible_paths = [
                self.checkpoint_path.parent / "scaler_params.pkl",
                self.checkpoint_path.parent / "scaler_ch4_soft.pkl",
                Path(Config.SCALER_PATH),
                Path(Ch4Config().CHECKPOINT_DIR) / "scaler_ch4_soft.pkl"
            ]

            for path in possible_paths:
                if path.exists():
                    target_path = self.output_dir / "scaler_params.pkl"
                    import shutil
                    shutil.copy(path, target_path)
                    print(f"    ✓ Scaler found and copied: {path.name}")
                    return

            print(f"    ⚠ Scaler not found, skipping...")

    def save_threshold(self, threshold_path: Optional[str] = None):
        """保存阈值参数（第三章模型）"""
        if not self.is_ch4:
            if threshold_path and os.path.exists(threshold_path):
                target_path = self.output_dir / "threshold.npy"
                import shutil
                shutil.copy(threshold_path, target_path)
                print(f"    ✓ Threshold copied to: {target_path}")
            else:
                # 尝试自动查找
                possible_paths = [
                    self.checkpoint_path.parent / "threshold.npy",
                    Path(Config.OUTPUT_DIR) / "threshold.npy"
                ]

                for path in possible_paths:
                    if path.exists():
                        target_path = self.output_dir / "threshold.npy"
                        import shutil
                        shutil.copy(path, target_path)
                        print(f"    ✓ Threshold found and copied")
                        return

                print(f"    ⚠ Threshold not found, skipping...")

    def save_fusion_params(self, fusion_params_path: Optional[str] = None):
        """保存融合参数（第三章模型）"""
        if not self.is_ch4:
            if fusion_params_path and os.path.exists(fusion_params_path):
                target_path = self.output_dir / "fusion_params.json"
                import shutil
                shutil.copy(fusion_params_path, target_path)
                print(f"    ✓ Fusion params copied to: {target_path}")
            else:
                # 尝试自动查找
                possible_paths = [
                    self.checkpoint_path.parent / "fusion_params.json",
                    Path(Config.FUSION_PARAMS_PATH)
                ]

                for path in possible_paths:
                    if path.exists():
                        target_path = self.output_dir / "fusion_params.json"
                        import shutil
                        shutil.copy(path, target_path)
                        print(f"    ✓ Fusion params found and copied")
                        return

                print(f"    ⚠ Fusion params not found, skipping...")

    def export(self,
               scaler_path: Optional[str] = None,
               threshold_path: Optional[str] = None,
               fusion_params_path: Optional[str] = None):
        """
        执行完整的导出流程
        """
        print(f"\n{'=' * 60}")
        print(f">>> Starting Export Process")
        print(f"{'=' * 60}\n")

        # 1. 加载模型
        model = self.load_model()

        # 2. 创建虚拟输入
        if self.is_ch4:
            dummy_inputs = self.create_dummy_inputs_ch4()
        else:
            dummy_inputs = self.create_dummy_inputs_ch3()

        # 2.5 验证模型forward pass
        print(f"\n>>> Validating model forward pass...")
        try:
            device = next(model.parameters()).device
            dummy_inputs_device = tuple(inp.to(device) for inp in dummy_inputs)
            with torch.no_grad():
                output = model(*dummy_inputs_device)
                if isinstance(output, tuple):
                    output = output[0]
                print(f"    ✓ Forward pass successful")
                print(f"    Output shape: {output.shape}")
        except Exception as e:
            print(f"    ⚠ Forward pass warning: {e}")
            print(f"    Continuing with export anyway...")

        # 3. 导出ONNX
        onnx_path = self.output_dir / "model.onnx"
        self.export_onnx(model, dummy_inputs, onnx_path)

        # 4. 保存辅助文件
        print(f"\n>>> Saving auxiliary files...")
        self.save_scaler(scaler_path)
        if not self.is_ch4:
            self.save_threshold(threshold_path)
            self.save_fusion_params(fusion_params_path)

        # 5. 生成导出摘要
        self._generate_summary(onnx_path)

        print(f"\n{'=' * 60}")
        print(f">>> Export Completed Successfully!")
        print(f"    Output Directory: {self.output_dir}")
        print(f"    ONNX Model: {onnx_path.name}")
        print(f"{'=' * 60}\n")

    def _generate_summary(self, onnx_path: Path):
        """生成导出摘要文件"""
        import datetime
        device_info = "CPU"
        if torch.cuda.is_available():
            try:
                device_info = torch.cuda.get_device_name(0)
            except:
                device_info = "CUDA Available"

        summary = {
            "model_name": self.model_name,
            "chapter": 4 if self.is_ch4 else 3,
            "checkpoint_path": str(self.checkpoint_path),
            "onnx_path": str(onnx_path),
            "export_timestamp": datetime.datetime.now().isoformat(),
            "device": device_info,
            "pytorch_version": torch.__version__,
        }

        summary_path = self.output_dir / "export_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"    ✓ Export summary saved to: {summary_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch models to ONNX for edge deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export Chapter 4 model (auto-detect)
  python export_to_edge.py --checkpoint checkpoints_ch4/rq1/Phys-RDLinear/model.pth --output edge_models

  # Export Chapter 3 model with explicit paths
  python export_to_edge.py \\
    --checkpoint checkpoints/Horizon_RDLinear_P24/checkpoint.pth \\
    --output edge_models/ch3 \\
    --scaler checkpoints/Horizon_RDLinear_P24/scaler_params.pkl \\
    --threshold checkpoints/Horizon_RDLinear_P24/threshold.npy

  # Specify model name explicitly
  python export_to_edge.py \\
    --checkpoint model.pth \\
    --output edge_models \\
    --model-name Phys-RDLinear
        """
    )

    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint (.pth file)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output directory for exported files'
    )

    parser.add_argument(
        '--model-name', '-m',
        type=str,
        default=None,
        help='Model name (auto-detected if not specified)'
    )

    parser.add_argument(
        '--scaler',
        type=str,
        default=None,
        help='Path to scaler file (auto-searched if not specified)'
    )

    parser.add_argument(
        '--threshold',
        type=str,
        default=None,
        help='Path to threshold file (Chapter 3 only, auto-searched if not specified)'
    )

    parser.add_argument(
        '--fusion-params',
        type=str,
        default=None,
        help='Path to fusion params file (Chapter 3 only, auto-searched if not specified)'
    )

    args = parser.parse_args()

    # 验证checkpoint存在
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)

    # 执行导出
    try:
        exporter = ModelExporter(
            checkpoint_path=args.checkpoint,
            output_dir=args.output,
            model_name=args.model_name
        )

        exporter.export(
            scaler_path=args.scaler,
            threshold_path=args.threshold,
            fusion_params_path=args.fusion_params
        )

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()


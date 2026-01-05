import os
import json
import onnx
import numpy as np
from pathlib import Path


def get_onnx_param_count(onnx_path):
    """
    加载ONNX模型并计算参数量 (权重数量)
    """
    try:
        # 加载模型 (不加载大模型数据以节省内存，只加载结构和权重元数据)
        model = onnx.load(onnx_path, load_external_data=False)

        count = 0
        # 遍历所有初始化器 (Initializers 存储了模型的权重)
        for tensor in model.graph.initializer:
            # 乘积计算该层参数数量 (例如 [64, 3, 7] -> 1344)
            dims = tensor.dims
            if len(dims) > 0:
                count += np.prod(dims)
            else:
                count += 1  # 标量
        return count
    except Exception as e:
        print(f"[Error] 无法读取 ONNX 文件: {e}")
        return None


def format_params(num):
    """将数字格式化为 K 或 M"""
    if num is None:
        return "N/A"
    if num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return str(num)


def scan_and_calculate(base_dir):
    base_path = Path(base_dir)
    results = []

    print(f"正在扫描并计算 ONNX 参数，根目录: {base_path} ...\n")

    # 递归查找所有 export_summary.json
    for json_file in base_path.rglob("export_summary.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            model_name = data.get('model_name', json_file.parent.name)
            rel_onnx_path = data.get('onnx_path', '')

            if not rel_onnx_path:
                continue

            # --- 路径拼接逻辑 ---
            # JSON里的路径可能是 "edge_models\ch4_Phys-RDLinear\model.onnx"
            # 我们需要把它转换成当前脚本能访问的绝对路径

            # 假设 base_dir 就是包含 edge_models 的那个大目录
            # 或者我们尝试相对于 json 文件的位置去寻找

            # 策略A: 假设 json 中的路径是相对于 base_dir 的根目录
            # 需要回溯到包含 edge_models 的层级。
            # 简单的做法：看 JSON 所在的文件夹里有没有 model.onnx

            local_onnx = json_file.parent / "model.onnx"
            target_onnx_path = None

            if local_onnx.exists():
                target_onnx_path = local_onnx
            else:
                # 尝试使用 json 里的路径拼接
                # 注意处理 Windows 的 \\ 分隔符
                clean_rel_path = rel_onnx_path.replace('\\', '/')
                target_onnx_path = base_path / clean_rel_path

                # 如果还是找不到，尝试在 base_dir 的上级找 (取决于你在哪里运行脚本)
                if not target_onnx_path.exists():
                    # 最后的兜底：假设 json 里的 edge_models 指的是当前结构
                    pass

            if target_onnx_path and target_onnx_path.exists():
                # 1. 计算文件大小 (MB)
                file_size_mb = target_onnx_path.stat().st_size / (1024 * 1024)

                # 2. 计算参数量
                params_count = get_onnx_param_count(str(target_onnx_path))

                results.append({
                    'name': model_name,
                    'path': str(json_file.parent),
                    'params': params_count,
                    'size_mb': file_size_mb
                })
            else:
                results.append({
                    'name': model_name,
                    'path': str(json_file.parent),
                    'params': None,
                    'size_mb': 0,
                    'error': 'ONNX missing'
                })

        except Exception as e:
            print(f"[Skip] {json_file}: {e}")

    # 排序
    results.sort(key=lambda x: x['params'] if x['params'] else 0)

    # 输出表格
    print(f"{'Model Name':<25} | {'Params':<10} | {'Size (MB)':<10} | {'Path (Folder)'}")
    print("-" * 90)
    for res in results:
        p_str = format_params(res['params'])
        s_str = f"{res['size_mb']:.2f} MB" if res['size_mb'] else "N/A"

        # 路径简化
        short_path = "..." + os.sep + os.path.basename(res['path'])

        if 'error' in res:
            print(f"{res['name']:<25} | {'Missing':<10} | {'N/A':<10} | {short_path} (File not found)")
        else:
            print(f"{res['name']:<25} | {p_str:<10} | {s_str:<10} | {short_path}")
    print("-" * 90)


if __name__ == "__main__":
    # 请修改为包含 edge_models 文件夹的根目录
    # 例如: D:\Darwin_base\...\Project_A_Fault_Diagnosis\chapter_3
    # 或者是它的上一级，只要能包含住 edge_models 即可

    TARGET_DIR = r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis"

    scan_and_calculate(TARGET_DIR)
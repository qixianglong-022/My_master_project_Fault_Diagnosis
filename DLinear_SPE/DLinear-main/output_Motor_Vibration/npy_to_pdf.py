import numpy as np
import matplotlib.pyplot as plt
import os


def visualize_anomaly_detection(data_dir, spe_name='test_spe.npy', thresh_name='test_threshold.npy'):
    """
    读取指定目录下的 SPE 和 Threshold .npy 文件并进行可视化对比。
    """

    # 1. 构建完整路径
    spe_path = os.path.join(data_dir, spe_name)
    thresh_path = os.path.join(data_dir, thresh_name)

    # 2. 读取数据
    print(f"正在读取目录: {data_dir}")

    # 读取 SPE
    if os.path.exists(spe_path):
        spe_data = np.load(spe_path)
        print(f"已加载 SPE 数据: {spe_name}, 形状: {spe_data.shape}")
    else:
        print(f"错误: 找不到文件 {spe_name}")
        return

    # 读取 Threshold (如果存在)
    has_threshold = False
    if os.path.exists(thresh_path):
        thresh_data = np.load(thresh_path)
        print(f"已加载 Threshold 数据: {thresh_name}, 形状: {thresh_data.shape}")
        has_threshold = True
    else:
        print(f"提示: 找不到阈值文件 {thresh_name}，将只绘制 SPE。")
        thresh_data = None

    # 3. 数据预处理 (降维，确保是 1D 数组)
    # 很多时候保存的数据形状是 (N, 1)，绘图需要 (N,)
    spe_data = np.squeeze(spe_data)
    if has_threshold:
        thresh_data = np.squeeze(thresh_data)
        # --- 修复标量阈值导致报错的问题 ---
        # 如果 thresh_data 是 0维数组（即标量，只有一个数）
        if thresh_data.ndim == 0:
            print(f"检测到阈值是固定常数: {thresh_data}, 将自动转换为水平线。")
            # 创建一个和 SPE 长度一样的全同数组，填满这个阈值
            thresh_val = float(thresh_data)  # 取出数值
            thresh_data = np.full_like(spe_data, thresh_val)
        # --------------------------------------

    # 4. 绘图
    plt.figure(figsize=(15, 6))

    # 绘制 SPE 曲线
    plt.plot(spe_data, label='SPE (Loss/Error)', color='#1f77b4', linewidth=1, alpha=0.8)

    # 绘制 阈值 曲线
    if has_threshold:
        # 确保长度一致，如果不一致则截取短的那个长度进行绘制（防止报错）
        min_len = min(len(spe_data), len(thresh_data))
        plt.plot(thresh_data[:min_len], label='Dynamic Threshold', color='#d62728', linestyle='--', linewidth=1.5)

        # 可选：填充超出阈值的区域（高亮异常）
        # 这里假设长度对齐
        if len(spe_data) == len(thresh_data):
            plt.fill_between(range(len(spe_data)), spe_data, thresh_data,
                             where=(spe_data > thresh_data),
                             interpolate=True, color='red', alpha=0.3, label='Anomaly Detected')

    plt.title(f'Fault Diagnosis Result: {spe_name} vs {thresh_name}', fontsize=14)
    plt.xlabel('Time Steps / Samples', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)

    # 5. 显示或保存
    plt.tight_layout()
    plt.show()

    # 如果你想保存图片，取消下面这行的注释
    # plt.savefig(os.path.join(data_dir, 'result_visualization.png'), dpi=300)


if __name__ == "__main__":
    # --- 配置区域 ---

    # 1. 这里填你 .npy 文件所在的文件夹路径
    # 根据你之前的报错信息，路径可能是在 output 文件夹里
    target_dir = r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\chapter_3\DLinear_SPE\DLinear-main\output"

    # 或者如果是在 runs 文件夹里，请修改上面的路径。
    # 如果你想测试当前目录，可以使用: target_dir = "."

    # 2. 调用函数
    #visualize_anomaly_detection(target_dir, 'test_spe.npy', 'test_threshold.npy')
    visualize_anomaly_detection(target_dir, 'final_inference_spe.npy', 'final_inference_labels.npy')
import pandas as pd
import numpy as np
import os

# ================= 配置区域 =================
# 正常数据的路径
NORMAL_FILE = (r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis"
               r"\data_chapter3\HH-0-3_Microphone1_3_test.csv")#改1
# 故障数据的路径
FAULT_FILE = (r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis"
               r"\data_chapter3\FB-0-3_Microphone1_3_test.csv")#改2
# 输出文件路径
OUTPUT_FILE = (r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis"
               r"\data_chapter3\Mixed_HH_FB_0-3_Microphone1_3_test.csv")#改3

# 你的时间戳列名 (通常是 'date')
DATE_COL = 'date'
# 采样频率 (例如 '10ms', '1s', '1h')
# 如果你的数据是无时间戳的纯索引，这个可以忽略
FREQ = '1min'


# ===========================================

def main():
    print(f"正在读取正常数据: {NORMAL_FILE} ...")
    df_normal = pd.read_csv(NORMAL_FILE)

    print(f"正在读取故障数据: {FAULT_FILE} ...")
    df_fault = pd.read_csv(FAULT_FILE)

    # 1. 简单的拼接 (上下堆叠)
    # ignore_index=True 重置索引，防止索引冲突
    print("正在拼接数据...")
    df_mixed = pd.concat([df_normal, df_fault], axis=0, ignore_index=True)

    # 2. 【关键步骤】重做连续的时间戳
    # 我们以正常数据的第一行时间为起点，按原来的频率生成新的连续时间轴
    if DATE_COL in df_mixed.columns:
        print("正在重置时间戳以保持连续性...")

        # 作为起点
        start_date = pd.to_datetime('2025-01-01 00:00:00')

        # 生成新的时间序列
        # length: 总行数
        # freq: 采样频率 (需根据你实际数据调整，比如 1小时'H', 15分钟'15T')
        # 注意：如果你的频率很难确定，也可以简单地把原有的时间戳丢掉，模型里设置 freq='h' 即可

        # 这里演示更通用的方法：
        # 如果你知道采样间隔（比如每行间隔1小时），用 date_range
        # new_dates = pd.date_range(start=start_date, periods=len(df_mixed), freq=FREQ)
        # df_mixed[DATE_COL] = new_dates

        # 【备选方案】如果不想纠结频率，最简单的欺骗模型的方法：
        # 很多时候模型并不真正在意绝对时间，只在意相对增量。
        # 如果你的 DLinear 设置了 freq='h' (默认)，它只关心年月日时的特征。
        # 最稳妥且简单的方法是：直接读取第一段的时间间隔，往后推。

        # 自动推断频率（如果正常数据够长）
        time_diff = pd.to_datetime(df_normal[DATE_COL].iloc[1]) - pd.to_datetime(df_normal[DATE_COL].iloc[0])
        print(f"检测到采样间隔约为: {time_diff}")

        new_dates = pd.date_range(start=start_date, periods=len(df_mixed), freq=time_diff)
        df_mixed[DATE_COL] = new_dates

    else:
        print(f"警告: 未找到列名 {DATE_COL}，跳过时间戳处理。")

    # 3. 添加标签列 (Ground Truth) - 可选，方便后续画图验证
    # 0 代表正常，1 代表故障
    # 我们假设 df_normal 全是 0，df_fault 全是 1
    labels = np.zeros(len(df_mixed), dtype=int)
    # 故障部分的索引开始位置
    fault_start_idx = len(df_normal)
    labels[fault_start_idx:] = 1

    # 把标签保存为一个单独的文件，或者作为一列加进去（看你需求）
    # 这里建议存为 npy 方便测试脚本直接读，不污染 CSV 给模型读
    save_dir = r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\data_chapter3"
    save_path = os.path.join(save_dir, "Mixed_HH_FB_0-3_Microphone1_3_test.npy")#改4

    # 若文件夹不存在，自动创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已自动创建文件夹: {save_dir}")

    # 保存并打印信息
    np.save(save_path, labels)
    print(f"标签文件已保存:  (故障起始点: {fault_start_idx})")

    # 4. 保存 CSV
    df_mixed.to_csv(OUTPUT_FILE, index=False)
    print(f"拼接完成！已保存至: {OUTPUT_FILE}")
    print(f"总数据量: {len(df_mixed)} 行 (正常: {len(df_normal)} + 故障: {len(df_fault)})")


if __name__ == '__main__':
    main()
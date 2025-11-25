import csv
import os
from datetime import datetime, timedelta


def txt_to_csv_split_by_ratio(input_path, output_dir, split_ratio=0.5):
    """
    读取txt文件，根据 split_ratio (比例) 将数据切分为两个csv文件：
    1. 前 split_ratio % 的数据 -> _train.csv
    2. 后 (1 - split_ratio) % 的数据 -> _test.csv
    """

    # --- 配置区域 ---
    INDEX_TIME_ORIGINAL = 0  # 原始txt中的时间列
    INDEX_VIBRATION = 7  # 改1
    fake_start_time = datetime(2025, 1, 1, 0, 0, 0)

    # 时间单位配置：'hours' 、'minutes' 'seconds'
    # 注意：你之前的代码是 hours，如果数据量大建议改为 seconds
    TIME_UNIT = 'minutes'
    # ----------------

    # 1. 第一遍扫描：计算有效数据总行数
    print(f"正在预扫描计算总行数: {input_path} ...")
    total_rows = 0
    data_found = False

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if "Time (seconds) and Data Channels" in line:
                data_found = True
                continue
            if data_found and line:
                total_rows += 1

    if total_rows == 0:
        print("错误：未找到有效数据行，请检查文件格式。")
        return

    # 计算分割点的行号 (向下取整)
    split_index = int(total_rows * split_ratio)
    print(f"总行数: {total_rows}")
    print(f"分割比例: {split_ratio} (前 {split_ratio * 100}% 为训练集)")
    print(f"分割断点: 第 {split_index} 行")

    # 2. 准备输出文件路径
    base_name = os.path.basename(input_path).replace(".txt", "")
    train_csv_path = os.path.join(output_dir, f"{base_name}_Motor_Vibration_train.csv")#改2
    test_csv_path = os.path.join(output_dir, f"{base_name}_Motor_Vibration_test.csv")#改3

    # 3. 第二遍扫描：正式写入
    print("开始写入数据...")
    data_found = False
    global_count = 0  # 全局计数器 (0, 1, 2...)

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                open(train_csv_path, 'w', newline='', encoding='utf-8') as f_train, \
                open(test_csv_path, 'w', newline='', encoding='utf-8') as f_test:

            writer_train = csv.writer(f_train)
            writer_test = csv.writer(f_test)

            header = ['date', 'Motor_Vibration']#改4
            writer_train.writerow(header)
            writer_test.writerow(header)

            for line in f_in:
                line = line.strip()

                # 定位头
                if "Time (seconds) and Data Channels" in line:
                    data_found = True
                    continue

                if not data_found or not line:
                    continue

                columns = line.split()

                try:
                    # --- 生成伪造时间戳 (保持连续) ---
                    if TIME_UNIT == 'hours':
                        current_fake_time = fake_start_time + timedelta(hours=global_count)
                    elif TIME_UNIT == 'minutes':  # <--- 新增这个判断分支
                        current_fake_time = fake_start_time + timedelta(minutes=global_count)
                    else:
                        # 默认为秒
                        current_fake_time = fake_start_time + timedelta(seconds=global_count)

                    time_str = current_fake_time.strftime("%Y-%m-%d %H:%M:%S")
                    vibration_val = columns[INDEX_VIBRATION]

                    # --- 核心切分逻辑 (基于行号 global_count) ---
                    if global_count < split_index:
                        # 写入训练集
                        writer_train.writerow([time_str, vibration_val])
                    else:
                        # 写入测试集
                        writer_test.writerow([time_str, vibration_val])

                    global_count += 1

                except (ValueError, IndexError):
                    continue

        print("-" * 30)
        print(f"处理完成！")
        print(f"训练集: {train_csv_path} | 行数: {split_index}")
        print(f"测试集: {test_csv_path}  | 行数: {total_rows - split_index}")

    except Exception as e:
        print(f"发生错误: {e}")


# --- 执行配置 ---
if __name__ == "__main__":
    input_dir = r"C:\毕业材料_齐祥龙\电机故障数据集\实验台数据采集\第1组——电机健康状态：健康（HH）"#改5
    output_dir = r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\data_chapter3"

    file_name = "HH-0-3.txt"#改6

    # 设置分割比例 (0.0 ~ 1.0)
    # 0.7 表示前 70% 是训练集，后 30% 是测试集
    RATIO = 0.5

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_full_path = os.path.join(input_dir, file_name)

    txt_to_csv_split_by_ratio(input_full_path, output_dir, split_ratio=RATIO)
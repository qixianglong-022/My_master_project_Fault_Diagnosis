import csv
import os
from datetime import datetime, timedelta  # 引入时间处理模块


def txt_to_csv_with_fake_time(input_path, output_path, time_limit=25.6):
    # 定义列索引
    INDEX_TIME_ORIGINAL = 0  # 原始txt中的时间列（仅用于判断何时停止）
    INDEX_VIBRATION = 8  # 电机振动列

    # 1. 定义起始伪造时间：2025-01-01 00:00:00
    fake_start_time = datetime(2025, 1, 1, 0, 0, 0)

    data_found = False
    count = 0

    print(f"正在读取: {input_path}")

    try:
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f_in, \
                open(output_path, 'w', newline='', encoding='utf-8') as f_out:

            writer = csv.writer(f_out)

            # --- 修改点 A：表头强制为 date ---
            writer.writerow(['date', 'Motor_Vibration'])

            for line in f_in:
                line = line.strip()

                if "Time (seconds) and Data Channels" in line:
                    data_found = True
                    continue

                if not data_found or not line:
                    continue

                columns = line.split()

                try:
                    # 获取原始时间，仅用于判断是否截断
                    original_time_val = float(columns[INDEX_TIME_ORIGINAL])

                    # 如果原始时间超过12.8秒，停止
                    if original_time_val > time_limit:
                        print(f"原始数据时间已达 {time_limit}s，停止处理。")
                        break

                    # --- 修改点 B：生成伪造时间戳 ---
                    # 逻辑：当前时间 = 起始时间 + (当前行数 * 1秒)
                    current_fake_time = fake_start_time + timedelta(hours=count)

                    # 格式化为字符串 "2025-01-01 00:00:00"
                    time_str = current_fake_time.strftime("%Y-%m-%d %H:%M:%S")

                    vibration_val = columns[INDEX_VIBRATION]

                    # 写入：[伪造时间, 振动值]
                    writer.writerow([time_str, vibration_val])
                    count += 1

                except (ValueError, IndexError):
                    continue

        print(f"处理成功！生成了 {count} 行数据。")
        print(f"伪造时间范围: {fake_start_time} 到 {fake_start_time + timedelta(seconds=count)}")
        print(f"保存至: {output_path}")

    except FileNotFoundError:
        print(f"错误：找不到文件 -> {input_path}")
    except Exception as e:
        print(f"发生错误: {e}")


# --- 执行配置 ---
if __name__ == "__main__":
    # 路径配置保持不变
    input_dir = r"C:\毕业材料_齐祥龙\电机故障数据集\实验台数据采集\第1组——电机健康状态：健康（HH）"
    output_dir = r"D:\Darwin_base\My_Knowledge_Base\5_项目实践 (Projects)\Project_A_Fault_Diagnosis\data_chapter3"

    # 请修改具体的txt文件名
    file_name = "HH-0-3.txt"  # <--- 修改这里

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_full_path = os.path.join(input_dir, file_name)
    output_file_name = file_name.replace(".txt", ".csv")
    output_full_path = os.path.join(output_dir, output_file_name)

    txt_to_csv_with_fake_time(input_full_path, output_full_path)
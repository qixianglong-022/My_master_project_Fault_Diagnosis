import os
import subprocess
import pandas as pd
import json

# 定义要对比的模型列表 (对应论文 Table 4.3)
MODELS = [
    'ResNet-18',
    'FD-CNN',
    'TiDE',
    'Vanilla RDLinear',
    'Phys-RDLinear'
]


def main():
    print("========================================================")
    print("   RQ1 Automation: Cross-Load Domain Generalization")
    print("========================================================")

    summary_list = []

    for model in MODELS:
        print(f"\n>>> [Auto] Processing Model: {model} ...")

        # 1. 调用 main_ch4.py 进行训练和评估
        cmd = f"python main_ch4.py --model_name \"{model}\" --gpu 0"
        ret = subprocess.call(cmd, shell=True)

        if ret != 0:
            print(f"[Error] Model {model} failed.")
            continue

        # 2. 读取结果
        result_path = f"checkpoints_ch4/{model}/summary_metrics.json"
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                res = json.load(f)
                summary_list.append(res)
        else:
            print(f"[Warn] Results not found for {model}")

    # 3. 生成最终表格
    if summary_list:
        df = pd.DataFrame(summary_list)
        # 调整列顺序
        cols = ['Model', 'Source_200kg', 'Target_0kg', 'Target_400kg', 'Avg_Target']
        df = df[cols]

        # 格式化保留1位小数
        df = df.round(1)

        print("\n========================================================")
        print("   Final Table (RQ1 Result)")
        print("========================================================")
        print(df.to_string(index=False))

        df.to_csv("Table_RQ1_Main_Results.csv", index=False)
        print(f"\n>>> Table saved to: Table_RQ1_Main_Results.csv")
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()
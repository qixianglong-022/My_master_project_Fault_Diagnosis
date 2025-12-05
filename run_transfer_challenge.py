# run_transfer_challenge.py
import pandas as pd
from config import Config
from eval_metrics import evaluate_current_task


def run_20_tasks():
    all_results = []

    # ================= 定义 20 组任务 =================
    tasks = []

    # 1. 同负载 (Load 2) - 转速迁移 (4组)
    # 30Hz(2), 60Hz(4), 30-60Hz(6), 60-30Hz(8)
    for s in ['2', '4', '6', '8']:
        tasks.append((f"A_SameLoad_Speed{s}", [('2', s)]))

    # 2. 变负载 (Load 0) - 全转速 (8组)
    for s in [str(i) for i in range(1, 9)]:
        tasks.append((f"B_Load0_Speed{s}", [('0', s)]))

    # 3. 变负载 (Load 4) - 全转速 (8组)
    for s in [str(i) for i in range(1, 9)]:
        tasks.append((f"C_Load4_Speed{s}", [('4', s)]))

    # ================= 循环执行 =================
    print(f">>> Starting Challenge: {len(tasks)} tasks in total.")

    for task_name, atoms in tasks:
        # 【核心动作】动态修改 Config
        Config.TEST_TASK_NAME = task_name
        Config.TEST_ATOMS = atoms

        # 调用评估函数
        # 注意：这里是在同一个进程里调用的，所以 Config 的修改会生效
        res = evaluate_current_task()

        if res:
            # 补充工况信息方便分析
            res['Load'] = atoms[0][0]
            res['Speed'] = atoms[0][1]
            all_results.append(res)

    # ================= 保存总表 =================
    df = pd.DataFrame(all_results)
    # 调整列顺序
    cols = ['Task', 'Load', 'Speed', 'AUC', 'F1', 'Recall', 'Accuracy', 'HH_Mean_SPE', 'FB_Mean_SPE']
    df = df[cols]

    df.to_excel("Final_Transfer_Report.xlsx", index=False)
    print("\n>>> All tasks finished. Report saved to Final_Transfer_Report.xlsx")


if __name__ == '__main__':
    run_20_tasks()
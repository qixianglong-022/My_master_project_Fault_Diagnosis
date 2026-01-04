import os
from pathlib import Path


def generate_tree(dir_path: Path, prefix: str = ""):
    """
    递归生成目录树结构
    :param dir_path: 当前需要遍历的目录 Path 对象
    :param prefix: 当前行的前缀字符串（用于控制缩进和连接线）
    """
    try:
        # 获取当前目录下所有文件和文件夹，并按名称排序
        # 过滤掉脚本自身，避免把自己也打印出来（可选）
        contents = list(dir_path.iterdir())
        contents.sort(key=lambda x: (not x.is_dir(), x.name.lower()))
    except PermissionError:
        print(f"{prefix}└── [权限不足]")
        return

    # 遍历当前目录下的项目
    pointers = [("├── ", "│   ")] * (len(contents) - 1) + [("└── ", "    ")]

    for pointer, path in zip(pointers, contents):
        connector, extension = pointer

        # 打印当前项
        print(f"{prefix}{connector}{path.name}")

        # 如果是目录，则递归进入
        if path.is_dir():
            generate_tree(path, prefix + extension)


def main():
    # 获取脚本所在的目录 (绝对路径)
    root_dir = Path(__file__).resolve().parent

    # 打印根目录名称
    print(f"{root_dir.name}/")

    # 开始生成树
    generate_tree(root_dir)


if __name__ == '__main__':
    main()
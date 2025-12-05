import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import Config
from utils.feature_extractor import FeatureExtractor


def parse_filename(fname):
    """
    解析文件名: "HH-0-1.txt" -> Load='0', Speed='1'
    """
    name_body = fname.replace('.txt', '').replace('.csv', '')
    parts = name_body.split('-')
    if len(parts) >= 3:
        load_id = parts[1]
        speed_id = parts[2]
        if load_id.isdigit() and speed_id.isdigit():
            return load_id, speed_id
    return None, None


def read_raw_data(file_path):
    """
    智能读取带复杂表头的 txt 文件
    """
    header_line_idx = None
    data_start_line = None
    column_names = None

    # 1. 预扫描：寻找表头和数据起点
    with open(file_path, 'r', encoding='gb18030', errors='ignore') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            # 找到包含列名的行 (Legend)
            if line.startswith('Legend'):
                # 假设是用制表符分隔的
                raw_cols = line.split('\t')
                # 清洗列名，去掉空字符串
                column_names = [c.strip() for c in raw_cols if c.strip()]
                header_line_idx = i

            # 找到数据开始前的标记行
            if line.startswith('Time') and 'Data Channels' in line:
                data_start_line = i + 1  # 数据从下一行开始
                break

    # 2. 兜底逻辑：如果没找到标记，尝试硬规则 (视具体情况而定)
    if data_start_line is None:
        # 如果文件很小或者格式不对，返回空
        if len(lines) < 20: return None
        data_start_line = 17  # 基于你提供的样本猜测

    if column_names is None:
        # 如果没找到 Legend，生成默认列名
        # 根据你提供的样本，应该是 27 列 (Time + 26 Channels)
        column_names = [f"Col_{i}" for i in range(27)]

    # 3. 使用 Pandas 读取数据块
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='gb18030',
            header=None,  # 不使用文件的 Header
            names=column_names,  # 使用我们解析出的列名
            skiprows=data_start_line,  # 跳过元数据行
            engine='python'  # Python 引擎更稳定
        )
        return df.values
    except Exception as e:
        print(f"    [Read Error] {e}")
        return None


def process_all():
    print(f">>> Root Data Dir: {Config.DATA_ROOT}")
    extractor = FeatureExtractor(Config)
    os.makedirs(Config.ATOMIC_DATA_DIR, exist_ok=True)

    for domain_code, folder_name in Config.DATA_DOMAINS.items():
        domain_path = os.path.join(Config.DATA_ROOT, folder_name)
        if not os.path.exists(domain_path):
            print(f"[Skip] Folder not found: {folder_name}")
            continue

        print(f"\n>>> Processing Domain: {domain_code} ...")
        files = [f for f in os.listdir(domain_path) if f.endswith('.txt')]

        success_count = 0

        for fname in tqdm(files, desc=f"Extracting {domain_code}"):
            load_id, speed_id = parse_filename(fname)
            if load_id is None: continue

            try:
                # === 读取数据 (使用新函数) ===
                file_path = os.path.join(domain_path, fname)
                data = read_raw_data(file_path)

                if data is None or data.shape[0] == 0:
                    print(f"    [Warn] Empty or invalid data: {fname}")
                    continue

                # === 提取原始信号 ===
                # 检查索引是否越界
                max_idx = data.shape[1] - 1
                required_indices = Config.COL_INDICES_VIB + Config.COL_INDICES_AUDIO + [Config.COL_INDEX_SPEED]
                if max(required_indices) > max_idx:
                    print(f"    [Error] Index out of bounds in {fname}. Max col: {max_idx}")
                    continue

                vib_raw = data[:, Config.COL_INDICES_VIB]
                speed_raw = data[:, Config.COL_INDEX_SPEED]
                # 声纹通道 (如果没有就全0)
                if max(Config.COL_INDICES_AUDIO) <= max_idx:
                    audio_raw = data[:, Config.COL_INDICES_AUDIO]
                else:
                    audio_raw = np.zeros((data.shape[0], 1))

                # === 特征提取 (同前) ===
                # 1. 振动
                vib_feats_list = []
                for i in range(vib_raw.shape[1]):
                    f = extractor.extract_vib_features(vib_raw[:, i])
                    vib_feats_list.append(f)
                vib_feats = np.concatenate(vib_feats_list, axis=1)

                # 2. 声纹 (Placeholder)
                n_frames = vib_feats.shape[0]
                audio_feats = np.zeros((n_frames, 13))
                # audio_feats = extractor.extract_audio_features(audio_raw[:, 0])

                # 3. 截断对齐
                min_len = min(len(vib_feats), len(audio_feats))

                # 4. 转速降采样
                speed_down = []
                for i in range(min_len):
                    s_idx = i * Config.HOP_LENGTH
                    e_idx = s_idx + Config.FRAME_SIZE
                    if e_idx > len(speed_raw): break
                    speed_down.append(np.mean(speed_raw[s_idx:e_idx]))

                # 再次对齐
                real_len = len(speed_down)
                final_x = np.concatenate([vib_feats[:real_len], audio_feats[:real_len]], axis=1)
                final_s = np.array(speed_down).reshape(-1, 1)

                # 保存
                save_name_x = f"{domain_code}_{load_id}_{speed_id}.npy"
                save_name_s = f"{domain_code}_{load_id}_{speed_id}_S.npy"
                np.save(os.path.join(Config.ATOMIC_DATA_DIR, save_name_x), final_x)
                np.save(os.path.join(Config.ATOMIC_DATA_DIR, save_name_s), final_s)

                success_count += 1

            except Exception as e:
                print(f"    [Error] Processing {fname}: {e}")
                # import traceback; traceback.print_exc() # Debug用

        print(f"    -> Processed {success_count} files.")


if __name__ == '__main__':
    process_all()
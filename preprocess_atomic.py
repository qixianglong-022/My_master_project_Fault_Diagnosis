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
                raw_cols = line.split('\t')
                column_names = [c.strip() for c in raw_cols if c.strip()]
                header_line_idx = i

            # 找到数据开始前的标记行
            if line.startswith('Time') and 'Data Channels' in line:
                data_start_line = i + 1  # 数据从下一行开始
                break

    # 2. 兜底逻辑
    if data_start_line is None:
        if len(lines) < 20: return None
        data_start_line = 17

    if column_names is None:
        column_names = [f"Col_{i}" for i in range(27)]

    # 3. Pandas 读取
    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            encoding='gb18030',
            header=None,
            names=column_names,
            skiprows=data_start_line,
            engine='python'
        )
        return df.values
    except Exception as e:
        print(f"    [Read Error] {e}")
        return None


def process_all():
    print(f">>> Root Data Dir: {Config.DATA_ROOT}")
    print(f">>> Output Dir:    {Config.ATOMIC_DATA_DIR}")

    extractor = FeatureExtractor(Config)
    os.makedirs(Config.ATOMIC_DATA_DIR, exist_ok=True)

    for domain_code, folder_name in Config.DATA_DOMAINS.items():
        domain_path = os.path.join(Config.DATA_ROOT, folder_name)
        if not os.path.exists(domain_path):
            print(f"[Skip] Folder not found: {folder_name}")
            continue

        print(f"\n>>> Processing Domain: {domain_code} ...")
        # 仅处理 txt 文件
        files = [f for f in os.listdir(domain_path) if f.endswith('.txt')]

        # 统计跳过和处理的数量
        skip_count = 0
        process_count = 0

        for fname in tqdm(files, desc=f"Extracting {domain_code}"):
            load_id, speed_id = parse_filename(fname)
            if load_id is None: continue

            # ================= [新增] 断点续传逻辑 =================
            save_name_x = f"{domain_code}_{load_id}_{speed_id}.npy"
            save_name_s = f"{domain_code}_{load_id}_{speed_id}_S.npy"
            path_x = os.path.join(Config.ATOMIC_DATA_DIR, save_name_x)
            path_s = os.path.join(Config.ATOMIC_DATA_DIR, save_name_s)

            # 如果两个文件都存在，直接跳过
            if os.path.exists(path_x) and os.path.exists(path_s):
                skip_count += 1
                continue
            # =======================================================

            try:
                # === 读取数据 ===
                file_path = os.path.join(domain_path, fname)
                data = read_raw_data(file_path)

                if data is None or data.shape[0] == 0:
                    print(f"    [Warn] Empty or invalid data: {fname}")
                    continue

                # === 提取原始信号 ===
                max_idx = data.shape[1] - 1
                required_indices = Config.COL_INDICES_VIB + Config.COL_INDICES_AUDIO + [Config.COL_INDEX_SPEED]
                if max(required_indices) > max_idx:
                    print(f"    [Error] Index out of bounds in {fname}. Max col: {max_idx}")
                    continue

                vib_raw = data[:, Config.COL_INDICES_VIB]
                speed_raw = data[:, Config.COL_INDEX_SPEED]

                # 声纹通道兼容性处理
                if max(Config.COL_INDICES_AUDIO) <= max_idx:
                    audio_raw = data[:, Config.COL_INDICES_AUDIO]
                else:
                    audio_raw = np.zeros((data.shape[0], len(Config.COL_INDICES_AUDIO)))

                # === 特征提取 (调用已修正的 FeatureExtractor) ===
                # 1. 振动 (不加窗的 RMS + Kurtosis)
                vib_feats_list = []
                for i in range(vib_raw.shape[1]):
                    f = extractor.extract_vib_features(vib_raw[:, i])
                    vib_feats_list.append(f)
                vib_feats = np.concatenate(vib_feats_list, axis=1)

                # 2. 声纹 (MFCC)
                # 注意：这里假设 Config.COL_INDICES_AUDIO 只有一个通道，如果有多个需循环
                if audio_raw.ndim == 1:
                    audio_feats = extractor.extract_audio_features(audio_raw)
                else:
                    audio_feats_list = []
                    for i in range(audio_raw.shape[1]):
                        f = extractor.extract_audio_features(audio_raw[:, i])
                        audio_feats_list.append(f)
                    audio_feats = np.concatenate(audio_feats_list, axis=1)

                # 3. 截断对齐
                min_len = min(len(vib_feats), len(audio_feats))

                # 4. 转速降采样 (取均值)
                speed_down = []
                for i in range(min_len):
                    s_idx = i * Config.HOP_LENGTH
                    e_idx = s_idx + Config.FRAME_SIZE
                    # 防止越界
                    if e_idx > len(speed_raw): break
                    speed_down.append(np.mean(speed_raw[s_idx:e_idx]))

                # 再次对齐 (因为 speed 采样可能也会少一点)
                real_len = len(speed_down)
                final_x = np.concatenate([vib_feats[:real_len], audio_feats[:real_len]], axis=1)
                final_s = np.array(speed_down).reshape(-1, 1)

                # === 保存 ===
                np.save(path_x, final_x)
                np.save(path_s, final_s)

                process_count += 1

            except Exception as e:
                print(f"    [Error] Processing {fname}: {e}")
                # import traceback; traceback.print_exc()

        print(f"    -> Domain {domain_code}: Processed {process_count}, Skipped {skip_count}")


if __name__ == '__main__':
    process_all()
import numpy as np
import librosa
import pyworld as pw
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import json
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

from project_config import get_config


def process_single_audio(wav_path, sampling_rate=22050, hop_length=256):
    """提取核心算法"""
    try:
        # 加载音频并强制重采样
        wav, _ = librosa.load(wav_path, sr=sampling_rate)
        wav_64 = wav.astype(np.float64)

        # 提取 Pitch
        frame_period = hop_length / sampling_rate * 1000
        _f0, t = pw.dio(wav_64, sampling_rate, frame_period=frame_period)
        f0 = pw.stonemask(wav_64, _f0, t, sampling_rate)

        # 提取 Energy
        S = librosa.magphase(librosa.stft(wav, n_fft=1024, hop_length=hop_length))[0]
        energy = np.linalg.norm(S, axis=0)

        # Pitch 插值
        if np.sum(f0 > 0) > 0:
            nonzero_ids = np.where(f0 > 0)[0]
            interp_fn = interp1d(nonzero_ids, f0[nonzero_ids],
                                 fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]), bounds_error=False)
            f0_interp = interp_fn(np.arange(len(f0)))
        else:
            f0_interp = f0

        # S-G 滤波
        win_len = 51 if len(f0_interp) > 51 else (len(f0_interp) // 2 * 2 - 1)
        if win_len < 3: win_len = 3

        pitch_sketch = savgol_filter(f0_interp, window_length=win_len, polyorder=2)
        energy_sketch = savgol_filter(energy, window_length=win_len, polyorder=2)

        def normalize(arr):
            v_min, v_max = np.min(arr), np.max(arr)
            if v_max - v_min < 1e-6: return np.zeros_like(arr).tolist()
            return ((arr - v_min) / (v_max - v_min)).tolist()

        return normalize(pitch_sketch), normalize(energy_sketch)
    except Exception:
        return None, None


def worker(wav_path, root_dir, output_root):
    """单个进程的任务单元"""
    try:
        # 计算相对路径，例如得到: 1h/Audio/01/S001T001E000N00000.wav
        relative_path = os.path.relpath(wav_path, root_dir)
        # 构造输出路径: E:/processed/1h/Audio/01/S001T001E000N00000.json
        output_json_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".json")

        # 自动递归创建子文件夹
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # 如果文件已存在则跳过（断点续传功能）
        if os.path.exists(output_json_path):
            return

        p_sketch, e_sketch = process_single_audio(wav_path)

        if p_sketch is not None:
            data = {
                "item_name": os.path.basename(wav_path),
                "pitch_sketch": p_sketch,
                "energy_sketch": e_sketch
            }
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f)  # 移除 indent 以减小 JSON 文件体积（18万个文件会节省很大空间）
    except Exception as e:
        pass


def batch_process_parallel(root_dir, output_root):
    # 规范化路径，处理 Windows 路径斜杠问题
    root_dir = os.path.abspath(root_dir)
    output_root = os.path.abspath(output_root)

    if not os.path.exists(root_dir):
        print(f"错误: 找不到输入目录 {root_dir}")
        return

    print(f"正在扫描目录: {root_dir} ...")
    wav_files = []
    # 递归扫描所有子目录下的 wav 文件
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    total_files = len(wav_files)
    if total_files == 0:
        print("未找到任何 .wav 文件，请确认 DATASET_DIR 路径是否指向 'after_catting' 这一层。")
        return

    print(f"共找到 {total_files} 个文件。开始多进程处理...")

    # 使用进程池利用多核 CPU
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(worker, root_dir=root_dir, output_root=output_root)
        list(tqdm(pool.imap_unordered(func, wav_files, chunksize=10), total=total_files))


if __name__ == "__main__":
    config = get_config()
    DATASET_DIR = str(config.project_data / "wav")
    OUTPUT_DIR = str(config.project_data / "processed_sketches_new")

    batch_process_parallel(DATASET_DIR, OUTPUT_DIR)

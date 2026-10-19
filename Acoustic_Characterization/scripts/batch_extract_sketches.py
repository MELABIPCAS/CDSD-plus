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
    """Extract pitch and energy sketches from one audio file."""
    try:
        # Load the audio and resample it to the target sampling rate.
        wav, _ = librosa.load(wav_path, sr=sampling_rate)
        wav_64 = wav.astype(np.float64)

        # Extract pitch.
        frame_period = hop_length / sampling_rate * 1000
        _f0, t = pw.dio(wav_64, sampling_rate, frame_period=frame_period)
        f0 = pw.stonemask(wav_64, _f0, t, sampling_rate)

        # Extract energy.
        S = librosa.magphase(librosa.stft(wav, n_fft=1024, hop_length=hop_length))[0]
        energy = np.linalg.norm(S, axis=0)

        # Interpolate pitch.
        if np.sum(f0 > 0) > 0:
            nonzero_ids = np.where(f0 > 0)[0]
            interp_fn = interp1d(nonzero_ids, f0[nonzero_ids],
                                 fill_value=(f0[nonzero_ids[0]], f0[nonzero_ids[-1]]), bounds_error=False)
            f0_interp = interp_fn(np.arange(len(f0)))
        else:
            f0_interp = f0

        # Apply Savitzky-Golay smoothing.
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
    """Process one audio file in a worker process."""
    try:
        # Compute the audio path relative to the input root.
        relative_path = os.path.relpath(wav_path, root_dir)
        # Construct the JSON output path under output_root.
        output_json_path = os.path.join(output_root, os.path.splitext(relative_path)[0] + ".json")

        # Create nested output directories as needed.
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # Skip existing output files to support resuming an interrupted run.
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
                json.dump(data, f)  # Omit indentation to reduce JSON file size for large batches.
    except Exception as e:
        pass


def batch_process_parallel(root_dir, output_root):
    # Normalize paths and handle Windows path separators.
    root_dir = os.path.abspath(root_dir)
    output_root = os.path.abspath(output_root)

    if not os.path.exists(root_dir):
        print(f"Error: input directory not found: {root_dir}")
        return

    print(f"Scanning directory: {root_dir} ...")
    wav_files = []
    # Recursively scan all subdirectories for WAV files.
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.wav'):
                wav_files.append(os.path.join(root, file))

    total_files = len(wav_files)
    if total_files == 0:
        print("No .wav files found. Check CDSD_PROJECT_DATA and its wav subdirectory.")
        return

    print(f"Found {total_files} files. Starting multiprocessing ...")

    # Use a process pool to utilize multiple CPU cores.
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        func = partial(worker, root_dir=root_dir, output_root=output_root)
        list(tqdm(pool.imap_unordered(func, wav_files, chunksize=10), total=total_files))


if __name__ == "__main__":
    config = get_config()
    DATASET_DIR = str(config.project_data / "wav")
    OUTPUT_DIR = str(config.project_data / "processed_sketches_new")

    batch_process_parallel(DATASET_DIR, OUTPUT_DIR)

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from project_config import get_config


CONFIG = get_config()
ROOT = CONFIG.output_root
PROJECT_DATA = CONFIG.project_data
META_PATH = ROOT / "data" / "segment_metadata_revised.csv"
OUTPUT_PATH = ROOT / "results" / "absolute_f0_features.csv"
TARGET_SR = 22050
HOP_LENGTH = 256
F0_FLOOR_HZ = 71.0
F0_CEIL_HZ = 800.0


def extract(row: tuple[str, str]) -> dict:
    segment_id, rel_path = row
    path = PROJECT_DATA.joinpath(*str(rel_path).replace("/", "\\").split("\\"))
    result = {
        "segment_id": segment_id,
        "f0_mean_hz": np.nan,
        "f0_std_hz": np.nan,
        "f0_range_hz": np.nan,
        "f0_cv": np.nan,
        "pitch_break_ratio_physical": np.nan,
        "f0_upper_bound_hit_rate": np.nan,
        "f0_extraction_status": "failed",
        "f0_failure_reason": "unreadable",
    }
    try:
        wav, source_sr = sf.read(path, dtype="float64", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if int(source_sr) != TARGET_SR:
            # For the dataset's 16 kHz source audio, 441/320 gives exactly 22.05 kHz.
            gcd = np.gcd(int(source_sr), TARGET_SR)
            wav = resample_poly(wav, TARGET_SR // gcd, int(source_sr) // gcd)
        wav64 = wav.astype(np.float64)
        frame_period_ms = HOP_LENGTH / TARGET_SR * 1000.0
        raw, time_axis = pw.dio(
            wav64, TARGET_SR, f0_floor=F0_FLOOR_HZ,
            f0_ceil=F0_CEIL_HZ, frame_period=frame_period_ms,
        )
        f0 = pw.stonemask(wav64, raw, time_axis, TARGET_SR)
        voiced = f0[f0 > 0]
        if len(voiced) == 0:
            result.update({"pitch_break_ratio_physical": 1.0, "f0_extraction_status": "no_voiced_frames", "f0_failure_reason": "no_voiced_frames"})
            return result
        mean = float(voiced.mean())
        result.update({
            "f0_mean_hz": mean,
            "f0_std_hz": float(voiced.std(ddof=0)),
            "f0_range_hz": float(voiced.max() - voiced.min()),
            "f0_cv": float(voiced.std(ddof=0) / mean),
            "pitch_break_ratio_physical": float(np.mean(f0 == 0)),
            "f0_upper_bound_hit_rate": float(np.mean(voiced >= F0_CEIL_HZ * 0.99)),
            "f0_extraction_status": "success",
            "f0_failure_reason": "none",
        })
    except Exception as exc:
        result["f0_failure_reason"] = type(exc).__name__
    return result


def main() -> None:
    meta = pd.read_csv(META_PATH, usecols=["segment_id", "wav_path"])
    rows = list(meta.itertuples(index=False, name=None))
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(extract, rows, chunksize=50), total=len(rows), desc="Absolute F0"))
    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(out["f0_extraction_status"].value_counts(dropna=False).to_dict())
    print({"segments_with_upper_bound_hits": int(out["f0_upper_bound_hit_rate"].fillna(0).gt(0).sum())})


if __name__ == "__main__":
    main()

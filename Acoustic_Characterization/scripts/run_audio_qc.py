from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import os

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from project_config import get_config


CONFIG = get_config()
ROOT = CONFIG.output_root
PROJECT_DATA = CONFIG.project_data
META_PATH = ROOT / "data" / "segment_metadata_revised.csv"
OUTPUT_PATH = ROOT / "results" / "audio_quality_control.csv"


def inspect_wav(row: tuple[str, str]) -> dict:
    segment_id, rel_path = row
    path = PROJECT_DATA.joinpath(*str(rel_path).replace("/", "\\").split("\\"))
    result = {
        "segment_id": segment_id,
        "audio_qc_status": "failed",
        "sample_rate_hz": np.nan,
        "channels": np.nan,
        "peak_dbfs": np.nan,
        "rms_dbfs": np.nan,
        "clipping_ratio": np.nan,
        "clipping_flag": False,
        "low_volume_flag": False,
        "audio_qc_failure_reason": "unreadable",
    }
    try:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        if data.size == 0:
            result["audio_qc_failure_reason"] = "empty_decoded_audio"
            return result
        absolute = np.abs(data)
        peak = float(absolute.max())
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
        clipping_ratio = float(np.mean(absolute >= 0.999))
        result.update({
            "audio_qc_status": "assessed",
            "sample_rate_hz": int(sr),
            "channels": int(data.shape[1]),
            "peak_dbfs": 20 * np.log10(max(peak, 1e-12)),
            "rms_dbfs": 20 * np.log10(max(rms, 1e-12)),
            "clipping_ratio": clipping_ratio,
            "clipping_flag": clipping_ratio >= 0.001,
            "low_volume_flag": rms < 10 ** (-45 / 20),
            "audio_qc_failure_reason": "none",
        })
    except Exception as exc:
        result["audio_qc_failure_reason"] = type(exc).__name__
    return result


def main() -> None:
    meta = pd.read_csv(META_PATH, usecols=["segment_id", "wav_path"])
    rows = list(meta.itertuples(index=False, name=None))
    workers = min(8, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(tqdm(pool.map(inspect_wav, rows, chunksize=100), total=len(rows), desc="Audio QC"))
    qc = pd.DataFrame(results)
    qc.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(qc["audio_qc_status"].value_counts(dropna=False).to_dict())
    print({"clipping": int(qc["clipping_flag"].sum()), "low_volume": int(qc["low_volume_flag"].sum())})


if __name__ == "__main__":
    main()

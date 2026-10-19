from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf


FRAME_LENGTH = 1024
HOP_LENGTH = 256
DBFS_FLOOR = 1e-12
PITCH_FLOOR_HZ = 71.0
PITCH_CEILING_HZ = 800.0


def resolve_project_path(project_root: Path, relative_path: str) -> Path:
    parts = str(relative_path).replace("/", "\\").split("\\")
    return project_root.joinpath(*(part for part in parts if part))


def _as_sketch_array(value: object, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name}_empty_or_not_1d")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}_nonfinite")
    if array.min() < -1e-8 or array.max() > 1.0 + 1e-8:
        raise ValueError(f"{name}_outside_0_1")
    return array


def _legacy_sketch_stats(array: np.ndarray) -> tuple[float, float, float, float]:
    nonzero = array[array > 0]
    if nonzero.size == 0:
        return 0.0, 0.0, float(np.ptp(array)), 0.0
    mean = float(np.mean(nonzero))
    std = float(np.std(nonzero, ddof=0))
    value_range = float(np.ptp(array))
    cv = std / mean if mean > 0 else 0.0
    return mean, std, value_range, cv


def extract_sketch_features(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "sketch_f0_mean_norm": np.nan,
        "sketch_f0_std_norm": np.nan,
        "sketch_f0_range_norm": np.nan,
        "sketch_f0_cv_norm": np.nan,
        "sketch_pitch_break_ratio": np.nan,
        "sketch_energy_mean_norm": np.nan,
        "sketch_energy_std_norm": np.nan,
        "sketch_energy_range_norm": np.nan,
        "sketch_f0_range_degenerate": False,
        "sketch_energy_range_degenerate": False,
        "sketch_status": "failed",
        "sketch_failure_reason": "unreadable",
    }
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        pitch = _as_sketch_array(payload["pitch_sketch"], "pitch_sketch")
        energy = _as_sketch_array(payload["energy_sketch"], "energy_sketch")
        f0_mean, f0_std, f0_range, f0_cv = _legacy_sketch_stats(pitch)
        energy_mean, energy_std, energy_range, _ = _legacy_sketch_stats(energy)
        result.update({
            "sketch_f0_mean_norm": f0_mean,
            "sketch_f0_std_norm": f0_std,
            "sketch_f0_range_norm": f0_range,
            "sketch_f0_cv_norm": f0_cv,
            "sketch_pitch_break_ratio": float(np.mean(pitch == 0)),
            "sketch_energy_mean_norm": energy_mean,
            "sketch_energy_std_norm": energy_std,
            "sketch_energy_range_norm": energy_range,
            "sketch_f0_range_degenerate": bool(np.isclose(f0_range, 1.0, atol=1e-8)),
            "sketch_energy_range_degenerate": bool(np.isclose(energy_range, 1.0, atol=1e-8)),
            "sketch_status": "success",
            "sketch_failure_reason": "none",
        })
    except Exception as exc:
        result["sketch_failure_reason"] = type(exc).__name__ + ":" + str(exc)
    return result


def _frame_rms_dbfs(audio: np.ndarray) -> tuple[float, float, float]:
    if audio.size < FRAME_LENGTH:
        raise ValueError("audio_shorter_than_frame")
    if float(np.max(np.abs(audio))) <= DBFS_FLOOR:
        raise ValueError("silent_audio")
    frame_count = 1 + (audio.size - FRAME_LENGTH) // HOP_LENGTH
    shape = (frame_count, FRAME_LENGTH)
    strides = (audio.strides[0] * HOP_LENGTH, audio.strides[0])
    frames = np.lib.stride_tricks.as_strided(audio, shape=shape, strides=strides)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    dbfs = 20.0 * np.log10(np.maximum(rms, DBFS_FLOOR))
    return float(np.mean(dbfs)), float(np.std(dbfs, ddof=0)), float(np.ptp(dbfs))


def _metric(name: str, function: Callable[[], float]) -> tuple[float, str, str | None]:
    try:
        value = float(function())
        if not np.isfinite(value):
            return np.nan, "failed", f"{name}:nonfinite"
        return value, "success", None
    except Exception as exc:
        return np.nan, "failed", f"{name}:{type(exc).__name__}"


def _blank_voice_result(reason: str) -> dict[str, object]:
    return {
        "energy_mean_dbfs": np.nan,
        "energy_std_db": np.nan,
        "energy_range_db": np.nan,
        "jitter": np.nan,
        "shimmer": np.nan,
        "hnr": np.nan,
        "energy_status": "failed",
        "jitter_status": "failed",
        "shimmer_status": "failed",
        "hnr_status": "failed",
        "extraction_status": "failed",
        "failure_reason": reason,
    }


def extract_energy_voice_quality(path: Path) -> dict[str, object]:
    try:
        data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    except Exception as exc:
        return _blank_voice_result(f"audio:{type(exc).__name__}")
    if data.size == 0:
        return _blank_voice_result("audio:empty_decoded_audio")

    mono = np.asarray(data.mean(axis=1), dtype=np.float64)
    if float(np.max(np.abs(mono))) <= DBFS_FLOOR:
        return _blank_voice_result("energy:silent_audio;voice:silent_audio")

    failures: list[str] = []
    result = _blank_voice_result("none")
    try:
        energy_mean, energy_std, energy_range = _frame_rms_dbfs(mono)
        result.update({
            "energy_mean_dbfs": energy_mean,
            "energy_std_db": energy_std,
            "energy_range_db": energy_range,
            "energy_status": "success",
        })
    except Exception as exc:
        result["energy_status"] = "failed"
        failures.append(f"energy:{str(exc)}")

    try:
        sound = parselmouth.Sound(mono, sampling_frequency=float(sample_rate))
        pitch = sound.to_pitch_ac(
            time_step=None,
            pitch_floor=PITCH_FLOOR_HZ,
            max_number_of_candidates=15,
            very_accurate=False,
            silence_threshold=0.03,
            voicing_threshold=0.45,
            octave_cost=0.01,
            octave_jump_cost=0.35,
            voiced_unvoiced_cost=0.14,
            pitch_ceiling=PITCH_CEILING_HZ,
        )
        pulses = call([sound, pitch], "To PointProcess (cc)")
        jitter, jitter_status, jitter_failure = _metric(
            "jitter", lambda: call(pulses, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        )
        shimmer, shimmer_status, shimmer_failure = _metric(
            "shimmer",
            lambda: call([sound, pulses], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6),
        )
        result.update({
            "jitter": jitter,
            "shimmer": shimmer,
            "jitter_status": jitter_status,
            "shimmer_status": shimmer_status,
        })
        failures.extend(x for x in (jitter_failure, shimmer_failure) if x)
    except Exception as exc:
        failures.extend([f"jitter:{type(exc).__name__}", f"shimmer:{type(exc).__name__}"])

    try:
        sound_for_hnr = parselmouth.Sound(mono, sampling_frequency=float(sample_rate))
        harmonicity = sound_for_hnr.to_harmonicity_cc(
            time_step=0.01,
            minimum_pitch=PITCH_FLOOR_HZ,
            silence_threshold=0.1,
            periods_per_window=1.0,
        )
        hnr, hnr_status, hnr_failure = _metric(
            "hnr", lambda: call(harmonicity, "Get mean", 0, 0)
        )
        result.update({"hnr": hnr, "hnr_status": hnr_status})
        if hnr_failure:
            failures.append(hnr_failure)
    except Exception as exc:
        failures.append(f"hnr:{type(exc).__name__}")

    statuses = [
        result["energy_status"], result["jitter_status"],
        result["shimmer_status"], result["hnr_status"],
    ]
    successes = statuses.count("success")
    result["extraction_status"] = "success" if successes == 4 else ("failed" if successes == 0 else "partial")
    result["failure_reason"] = "none" if not failures else ";".join(sorted(set(failures)))
    return result

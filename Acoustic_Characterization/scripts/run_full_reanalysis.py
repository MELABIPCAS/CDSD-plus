from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd
from project_config import get_config


CONFIG = get_config()
REPO_ROOT = CONFIG.repo_root
ROOT = CONFIG.output_root
SCRIPTS = REPO_ROOT / "scripts"
RESULTS = ROOT / "results"
LOGS = RESULTS / "logs"
MANIFEST = RESULTS / "run_manifest.json"

STAGE_ORDER = ["bootstrap", "audio_qc", "absolute_f0", "sketch", "energy_voice", "final"]
EXPECTED_OUTPUTS = {
    "bootstrap": ROOT / "data" / "segment_metadata_revised.csv",
    "audio_qc": RESULTS / "audio_quality_control.csv",
    "absolute_f0": RESULTS / "absolute_f0_features.csv",
    "sketch": RESULTS / "sketch_features.csv",
    "energy_voice": RESULTS / "energy_voice_quality_features.csv",
    "final": ROOT / "data" / "speaker_features_revised.csv",
}


def build_commands(python_executable: str) -> dict[str, list[str]]:
    return {
        "bootstrap": [python_executable, str(SCRIPTS / "run_acoustic_revision.py"), "--bootstrap-only"],
        "audio_qc": [python_executable, str(SCRIPTS / "run_audio_qc.py")],
        "absolute_f0": [python_executable, str(SCRIPTS / "run_absolute_f0.py")],
        "sketch": [python_executable, str(SCRIPTS / "run_sketch_features.py")],
        "energy_voice": [python_executable, str(SCRIPTS / "run_energy_voice_quality.py")],
        "final": [python_executable, str(SCRIPTS / "run_acoustic_revision.py")],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_output(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return summary
    summary.update({"bytes": path.stat().st_size, "sha256": sha256(path)})
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        summary["rows"] = len(frame)
        summary["columns"] = list(frame.columns)
        if "segment_id" in frame.columns:
            summary["unique_segment_id"] = int(frame["segment_id"].nunique())
        if "speaker_id" in frame.columns:
            summary["unique_speaker_id"] = int(frame["speaker_id"].nunique())
        for status_column in ["sketch_status", "f0_extraction_status", "extraction_status", "feature_extraction_status"]:
            if status_column in frame.columns:
                summary[status_column] = frame[status_column].value_counts(dropna=False).to_dict()
    return summary


def environment() -> dict[str, object]:
    packages = [
        "pandas", "numpy", "scipy", "librosa", "pyworld", "praat-parselmouth",
        "scikit-learn", "seaborn", "matplotlib", "soundfile", "openpyxl",
    ]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def run_pipeline(from_stage: str, dry_run: bool = False) -> dict[str, object]:
    commands = build_commands(sys.executable)
    start_index = STAGE_ORDER.index(from_stage)
    selected = STAGE_ORDER[start_index:]
    if dry_run:
        return {stage: commands[stage] for stage in selected}

    LOGS.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = LOGS / f"{timestamp}.log"
    manifest: dict[str, object] = {
        "verification_status": "ANALYZED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment(),
        "parameters": {
            "sketch_domain": "per-segment min-max normalized [0,1]",
            "energy_frame_length": 1024,
            "energy_hop_length": 256,
            "energy_center": False,
            "energy_unit": "dBFS",
            "pitch_floor_hz": 71.0,
            "pitch_ceiling_hz": 800.0,
            "jitter_period_floor_s": 0.0001,
            "jitter_period_ceiling_s": 0.02,
            "maximum_period_factor": 1.3,
            "maximum_amplitude_factor": 1.6,
            "hnr_time_step_s": 0.01,
            "hnr_silence_threshold": 0.1,
            "hnr_periods_per_window": 1.0,
        },
        "stages": [],
    }
    with log_path.open("w", encoding="utf-8") as log:
        for stage in selected:
            command = commands[stage]
            stage_record: dict[str, object] = {
                "stage": stage,
                "command": command,
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            log.write(f"\n[{stage}] {' '.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log.write(line)
                log.flush()
            return_code = process.wait()
            stage_record["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
            stage_record["exit_code"] = return_code
            stage_record["output"] = summarize_output(EXPECTED_OUTPUTS[stage])
            manifest["stages"].append(stage_record)
            MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if return_code != 0:
                manifest["status"] = "failed"
                manifest["failed_stage"] = stage
                MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                raise RuntimeError(f"stage failed: {stage} (exit code {return_code})")

    manifest["status"] = "completed"
    manifest["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["log_path"] = str(log_path)
    manifest["key_outputs"] = {
        str(path.relative_to(ROOT)): summarize_output(path)
        for path in [
            ROOT / "data" / "segment_metadata_revised.csv",
            ROOT / "data" / "segment_features_revised.csv",
            ROOT / "data" / "speaker_features_revised.csv",
            RESULTS / "severity_results_revised.csv",
            RESULTS / "sketch_features.csv",
            RESULTS / "absolute_f0_features.csv",
            RESULTS / "energy_voice_quality_features.csv",
        ]
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete dual-track acoustic reanalysis.")
    parser.add_argument("--from-stage", choices=STAGE_ORDER, default="bootstrap")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_pipeline(args.from_stage, args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

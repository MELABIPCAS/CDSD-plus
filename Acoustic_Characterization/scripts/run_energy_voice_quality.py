from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from acoustic_feature_utils import extract_energy_voice_quality, resolve_project_path
from project_config import get_config


CONFIG = get_config()
ROOT = CONFIG.output_root
DEFAULT_METADATA = ROOT / "data" / "segment_metadata_revised.csv"
DEFAULT_PROJECT_ROOT = CONFIG.project_data
DEFAULT_OUTPUT = ROOT / "results" / "energy_voice_quality_features.csv"
DEFAULT_CHECKPOINT = ROOT / "results" / "energy_voice_quality_features.checkpoint.csv"

OUTPUT_COLUMNS = [
    "segment_id",
    "energy_mean_dbfs", "energy_std_db", "energy_range_db",
    "jitter", "shimmer", "hnr",
    "energy_status", "jitter_status", "shimmer_status", "hnr_status",
    "extraction_status", "failure_reason",
]


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "wav_path"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"missing metadata columns: {sorted(missing)}")
    if metadata["segment_id"].duplicated().any():
        raise ValueError("duplicate segment_id in metadata")
    return metadata[["segment_id", "wav_path"]].sort_values("segment_id").reset_index(drop=True)


def _extract_row(task: tuple[str, str, str]) -> dict[str, object]:
    segment_id, relative_path, project_root = task
    path = resolve_project_path(Path(project_root), relative_path)
    return {"segment_id": segment_id, **extract_energy_voice_quality(path)}


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def _load_prior(output_path: Path, checkpoint_path: Path) -> pd.DataFrame:
    source = checkpoint_path if checkpoint_path.exists() else output_path
    if not source.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    prior = pd.read_csv(source)
    missing = set(OUTPUT_COLUMNS) - set(prior.columns)
    if missing:
        raise ValueError(f"checkpoint missing columns: {sorted(missing)}")
    if prior["segment_id"].duplicated().any():
        raise ValueError("duplicate segment_id in checkpoint")
    return prior[OUTPUT_COLUMNS].copy()


def run(
    metadata_path: Path,
    project_root: Path,
    output_path: Path,
    checkpoint_path: Path,
    *,
    workers: int | None = None,
    checkpoint_every: int = 1000,
    retry_failed: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    metadata = _validate_metadata(pd.read_csv(metadata_path))
    if limit is not None:
        metadata = metadata.head(limit)
    selected_ids = set(metadata["segment_id"])
    prior = _load_prior(output_path, checkpoint_path)
    prior = prior.loc[prior["segment_id"].isin(selected_ids)].copy()
    if retry_failed:
        prior = prior.loc[~prior["extraction_status"].eq("failed")].copy()
    completed = set(prior["segment_id"])
    pending = metadata.loc[~metadata["segment_id"].isin(completed)].copy()
    tasks = [(row.segment_id, row.wav_path, str(project_root)) for row in pending.itertuples(index=False)]

    collected = prior.to_dict("records")
    worker_count = workers or min(8, os.cpu_count() or 1)
    if worker_count == 1:
        iterator = map(_extract_row, tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Energy and voice quality"):
            collected.append(result)
            if checkpoint_every > 0 and len(collected) % checkpoint_every == 0:
                current = pd.DataFrame(collected, columns=OUTPUT_COLUMNS).sort_values("segment_id")
                _atomic_write(current, checkpoint_path)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            iterator = pool.map(_extract_row, tasks, chunksize=20)
            for result in tqdm(iterator, total=len(tasks), desc="Energy and voice quality"):
                collected.append(result)
                if checkpoint_every > 0 and len(collected) % checkpoint_every == 0:
                    current = pd.DataFrame(collected, columns=OUTPUT_COLUMNS).sort_values("segment_id")
                    _atomic_write(current, checkpoint_path)

    output = pd.DataFrame(collected, columns=OUTPUT_COLUMNS).sort_values("segment_id").reset_index(drop=True)
    if output["segment_id"].duplicated().any():
        raise ValueError("duplicate segment_id after extraction")
    if set(output["segment_id"]) != selected_ids:
        missing = sorted(selected_ids - set(output["segment_id"]))[:10]
        extra = sorted(set(output["segment_id"]) - selected_ids)[:10]
        raise ValueError(f"output ID mismatch; missing={missing}, extra={extra}")
    _atomic_write(output, checkpoint_path)
    _atomic_write(output, output_path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WAV energy, jitter, shimmer, and HNR.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(
        args.metadata, args.project_root, args.output, args.checkpoint,
        workers=args.workers, checkpoint_every=args.checkpoint_every,
        retry_failed=args.retry_failed, limit=args.limit,
    )
    print({
        "rows": len(output),
        "unique_segment_id": output["segment_id"].nunique(),
        "status": output["extraction_status"].value_counts(dropna=False).to_dict(),
    })


if __name__ == "__main__":
    main()

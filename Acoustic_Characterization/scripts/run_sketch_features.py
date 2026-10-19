from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from acoustic_feature_utils import extract_sketch_features, resolve_project_path
from project_config import get_config


CONFIG = get_config()
ROOT = CONFIG.output_root
DEFAULT_METADATA = ROOT / "data" / "segment_metadata_revised.csv"
DEFAULT_PROJECT_ROOT = CONFIG.project_data
DEFAULT_OUTPUT = ROOT / "results" / "sketch_features.csv"

OUTPUT_COLUMNS = [
    "segment_id",
    "sketch_f0_mean_norm", "sketch_f0_std_norm", "sketch_f0_range_norm", "sketch_f0_cv_norm",
    "sketch_pitch_break_ratio",
    "sketch_energy_mean_norm", "sketch_energy_std_norm", "sketch_energy_range_norm",
    "sketch_f0_range_degenerate", "sketch_energy_range_degenerate",
    "sketch_status", "sketch_failure_reason",
]


def _validate_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {"segment_id", "pitch_sketch_path", "energy_sketch_path"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"missing metadata columns: {sorted(missing)}")
    if metadata["segment_id"].duplicated().any():
        raise ValueError("duplicate segment_id in metadata")
    return metadata[list(required)].sort_values("segment_id").reset_index(drop=True)


def run(
    metadata_path: Path,
    project_root: Path,
    output_path: Path,
    limit: int | None = None,
) -> pd.DataFrame:
    metadata = _validate_metadata(pd.read_csv(metadata_path))
    if limit is not None:
        metadata = metadata.head(limit)

    rows: list[dict[str, object]] = []
    for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Sketch features"):
        pitch_path = resolve_project_path(project_root, row.pitch_sketch_path)
        energy_path = resolve_project_path(project_root, row.energy_sketch_path)
        if pitch_path != energy_path:
            result = extract_sketch_features(pitch_path)
            if result["sketch_status"] == "success":
                result["sketch_status"] = "failed"
                result["sketch_failure_reason"] = "separate_pitch_energy_json_paths"
        else:
            result = extract_sketch_features(pitch_path)
        rows.append({"segment_id": row.segment_id, **result})

    output = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values("segment_id").reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract normalized DrawSpeech sketch-domain features.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run(args.metadata, args.project_root, args.output, args.limit)
    print({
        "rows": len(output),
        "unique_segment_id": output["segment_id"].nunique(),
        "status": output["sketch_status"].value_counts(dropna=False).to_dict(),
    })


if __name__ == "__main__":
    main()

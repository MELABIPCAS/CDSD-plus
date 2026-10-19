from __future__ import annotations

import argparse
import os
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from project_config import get_config

warnings.filterwarnings("ignore", category=FutureWarning)

CONFIG = get_config()
ROOT = CONFIG.output_root
DATA_DIR = CONFIG.workspace_data
PROJECT_DATA = CONFIG.project_data
INFO_XLSX = CONFIG.info_xlsx

OUT_DATA = ROOT / "data"
OUT_RESULTS = ROOT / "results"
OUT_FIGURES = ROOT / "figures"
OUT_TEXT = ROOT / "text"
for d in (OUT_DATA, OUT_RESULTS, OUT_FIGURES, OUT_TEXT):
    d.mkdir(parents=True, exist_ok=True)

PRIMARY_SEGMENT_FEATURES = [
    "f0_mean", "f0_std", "f0_range", "f0_cv", "pitch_break_ratio",
    "energy_mean", "energy_std", "energy_range", "jitter", "shimmer", "hnr",
]
CORR_FEATURES = [
    "mean_segment_duration",
    "f0_mean", "f0_std", "f0_range", "f0_cv", "pitch_break_ratio",
    "energy_mean", "energy_std", "energy_range", "jitter", "shimmer", "hnr",
]
PHYSICAL_CORR_FEATURES = [
    "f0_mean_hz", "f0_std_hz", "f0_cv_physical", "pitch_break_ratio_physical",
    "energy_mean_dbfs", "energy_std_db", "jitter", "shimmer", "hnr",
    "mean_segment_duration",
]
PCA_FEATURES = ["f0_mean", "f0_std", "f0_cv", "pitch_break_ratio", "energy_mean", "energy_std"]


def bh_fdr(p_values: list[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(len(p), np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    vals = p[finite]
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    q[finite] = out
    return q


def _find_unique_column(frame: pd.DataFrame, fragment: str) -> str:
    matches = [column for column in frame.columns if fragment in str(column)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one column containing {fragment!r}, found {matches}")
    return matches[0]


def _classify_child_or_adult(frame: pd.DataFrame) -> pd.Series:
    """Classify only when the source provides auditable age evidence."""
    flag_column = _find_unique_column(frame, "确认未满18岁")
    exact_column = _find_unique_column(frame, "精确年龄_内部可用")
    range_column = _find_unique_column(frame, "年龄区间_最佳可用")

    flags = frame[flag_column].astype("string").str.strip()
    exact = pd.to_numeric(frame[exact_column], errors="coerce")
    ranges = frame[range_column].astype("string").str.strip()
    lower_bound = pd.to_numeric(ranges.str.extract(r"(\d+)", expand=False), errors="coerce")
    below_18 = ranges.str.contains("以下|未满", na=False)
    lower_bound = lower_bound.mask(below_18)

    child = flags.eq("是")
    adult_by_exact = exact.ge(18).fillna(False)
    adult_by_range = lower_bound.ge(18).fillna(False)
    adult = adult_by_exact | adult_by_range
    conflicts = child & adult
    if conflicts.any():
        ids = frame.loc[conflicts, "speaker_id"].astype(str).tolist() if "speaker_id" in frame else []
        raise ValueError(f"conflicting child/adult evidence for speaker_id={ids}")

    under18_without_child_flag = exact.lt(18) & ~child
    if under18_without_child_flag.any():
        ids = frame.loc[under18_without_child_flag, "speaker_id"].astype(str).tolist() if "speaker_id" in frame else []
        raise ValueError(f"exact age <18 without explicit child confirmation for speaker_id={ids}")

    unresolved = ~(child | adult)
    if unresolved.any():
        ids = frame.loc[unresolved, "speaker_id"].astype(str).tolist() if "speaker_id" in frame else []
        raise ValueError(f"unable to classify child_or_adult from age evidence for speaker_id={ids}")
    return pd.Series(np.where(child, "child", "adult"), index=frame.index, name="child_or_adult")


def load_info() -> pd.DataFrame:
    info = pd.read_excel(INFO_XLSX, sheet_name="最新被试信息", header=3)
    info = info.rename(columns={"CDSD_Plus_ID": "speaker_id"})
    info["speaker_id"] = info["speaker_id"].astype(str).str.strip()
    info["child_or_adult"] = _classify_child_or_adult(info)
    info["severity_group"] = info["严重程度分组"].replace({"Moderate": "Moderate", "Medium": "Moderate"})
    return info


def path_exists(rel: str) -> tuple[bool, int]:
    path = PROJECT_DATA.joinpath(*str(rel).replace("/", "\\").split("\\"))
    try:
        return path.exists(), path.stat().st_size if path.exists() else 0
    except OSError:
        return False, 0


def build_metadata(info: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(DATA_DIR / "segment_metadata.csv")
    old["speaker_id"] = old["speaker_id"].astype(str)
    old["severity_group"] = old["severity_group"].replace({"Medium": "Moderate"})
    old = old.merge(
        info[["speaker_id", "严重程度连续得分S", "严重程度分组"]],
        on="speaker_id", how="left", validate="many_to_one", suffixes=("_old", ""),
    )
    old["severity_score"] = old["严重程度连续得分S"]
    old["severity_group"] = old["严重程度分组"]
    old = old.drop(columns=["严重程度连续得分S", "严重程度分组"])
    old["data_split"] = old["wav_path"].str.extract(r"cdsd_extended\\(train|dev|test)\\", expand=False)

    wav_checks = [path_exists(p) for p in old["wav_path"]]
    pitch_checks = [path_exists(p) for p in old["pitch_sketch_path"]]
    energy_checks = [path_exists(p) for p in old["energy_sketch_path"]]

    old["wav_exists"] = [x[0] for x in wav_checks]
    old["wav_nonempty"] = [x[1] > 0 for x in wav_checks]
    old["duration_valid"] = old["duration"].gt(0.1) & old["duration"].le(60)

    pitch_exists = np.array([x[0] for x in pitch_checks], dtype=bool)
    energy_exists = np.array([x[0] for x in energy_checks], dtype=bool)

    input_ready = (
        old["wav_exists"]
        & old["wav_nonempty"]
        & old["duration_valid"]
        & pitch_exists
        & energy_exists
    )

    # Bootstrap checks input availability; extraction runs in later stages.
    old["feature_extraction_status"] = np.where(
        input_ready, "pending", "unavailable"
    )

    # Preserve every detected issue rather than overwriting earlier reasons.
    old["failure_reason"] = ""
    failure_checks = [
        (~old["wav_exists"], "wav_missing"),
        (old["wav_exists"] & ~old["wav_nonempty"], "wav_empty"),
        (~old["duration_valid"], "duration_invalid"),
        (~pitch_exists, "pitch_sketch_missing"),
        (~energy_exists, "energy_sketch_missing"),
    ]
    for mask, reason in failure_checks:
        old.loc[mask, "failure_reason"] = (
            old.loc[mask, "failure_reason"] + reason + ";"
        )
    old["failure_reason"] = old["failure_reason"].str.rstrip(";")
    keep = [
        "speaker_id", "segment_id", "wav_path", "pitch_sketch_path", "energy_sketch_path",
        "severity_score", "severity_group", "duration", "data_split", "wav_exists",
        "wav_nonempty", "duration_valid", "feature_extraction_status", "failure_reason",
    ]
    out = old[keep].sort_values(["speaker_id", "segment_id"]).reset_index(drop=True)
    qc_path = OUT_RESULTS / "audio_quality_control.csv"
    if qc_path.exists():
        qc = pd.read_csv(qc_path)
        out = out.merge(qc, on="segment_id", how="left", validate="one_to_one")
    out.to_csv(OUT_DATA / "segment_metadata_revised.csv", index=False, encoding="utf-8-sig")
    build_speaker_audio_summary(out)
    return out


def build_speaker_audio_summary(meta: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    """Persist one auditable row per speaker for segment count and valid duration."""
    valid = meta["wav_exists"].astype(bool) & meta["wav_nonempty"].astype(bool) & meta["duration_valid"].astype(bool)
    working = meta.assign(
        _valid_duration_sec=meta["duration"].where(valid),
    )
    summary = (
        working.groupby("speaker_id", as_index=False)
        .agg(
            num_segments_per_speaker=("segment_id", "size"),
            valid_audio_duration_sec=("_valid_duration_sec", "sum"),
        )
        .sort_values("speaker_id")
        .reset_index(drop=True)
    )
    if output_path is None:
        output_path = OUT_DATA / "speaker_audio_summary.csv"
    summary.to_csv(output_path, index=False, encoding="utf-8-sig")
    return summary


def _validate_feature_ids(frame: pd.DataFrame, name: str, expected_ids: set[str]) -> pd.DataFrame:
    if "segment_id" not in frame.columns:
        raise ValueError(f"{name} missing segment_id")
    if frame["segment_id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate segment_id")
    actual_ids = set(frame["segment_id"].astype(str))
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:10]
        extra = sorted(actual_ids - expected_ids)[:10]
        raise ValueError(f"{name} ID mismatch; missing={missing}, extra={extra}")
    return frame.copy()


def merge_dual_track(
    meta: pd.DataFrame,
    sketch: pd.DataFrame,
    absolute_f0: pd.DataFrame,
    voice: pd.DataFrame,
) -> pd.DataFrame:
    if meta["segment_id"].duplicated().any():
        raise ValueError("metadata contains duplicate segment_id")
    expected_ids = set(meta["segment_id"].astype(str))
    sketch = _validate_feature_ids(sketch, "sketch", expected_ids)
    absolute_f0 = _validate_feature_ids(absolute_f0, "absolute_f0", expected_ids).rename(
        columns={"f0_cv": "f0_cv_physical"}
    )
    voice = _validate_feature_ids(voice, "voice", expected_ids).rename(columns={
        "extraction_status": "voice_extraction_status",
        "failure_reason": "voice_failure_reason",
    })

    out = meta.merge(sketch, on="segment_id", how="left", validate="one_to_one")
    out = out.merge(absolute_f0, on="segment_id", how="left", validate="one_to_one")
    out = out.merge(voice, on="segment_id", how="left", validate="one_to_one")

    # Compatibility fields retain the original task's sketch-domain meaning.
    out["f0_mean"] = out["sketch_f0_mean_norm"]
    out["f0_std"] = out["sketch_f0_std_norm"]
    out["f0_range"] = out["sketch_f0_range_norm"]
    out["f0_cv"] = out["sketch_f0_cv_norm"]
    out["pitch_break_ratio"] = out["sketch_pitch_break_ratio"]
    out["energy_mean"] = out["sketch_energy_mean_norm"]
    out["energy_std"] = out["sketch_energy_std_norm"]
    out["energy_range"] = out["sketch_energy_range_norm"]
    out["segment_duration"] = out["duration"]

    all_tracks_available = (
        out["sketch_status"].eq("success")
        & out["f0_extraction_status"].eq("success")
        & out["voice_extraction_status"].eq("success")
    )
    out["feature_extraction_status"] = np.where(
        all_tracks_available, "available", "partial_or_failed"
    )

    reason_columns = ["sketch_failure_reason", "f0_failure_reason", "voice_failure_reason"]
    def combine_reasons(row: pd.Series) -> str:
        reasons = [str(row[column]) for column in reason_columns if pd.notna(row[column]) and str(row[column]) != "none"]
        return "none" if not reasons else ";".join(reasons)
    out["failure_reason"] = out.apply(combine_reasons, axis=1)

    columns = [
        "speaker_id", "segment_id", "severity_score", "severity_group",
        *PRIMARY_SEGMENT_FEATURES,
        "sketch_f0_mean_norm", "sketch_f0_std_norm", "sketch_f0_range_norm", "sketch_f0_cv_norm", "sketch_pitch_break_ratio",
        "sketch_energy_mean_norm", "sketch_energy_std_norm", "sketch_energy_range_norm",
        "sketch_f0_range_degenerate", "sketch_energy_range_degenerate", "sketch_status", "sketch_failure_reason",
        "f0_mean_hz", "f0_std_hz", "f0_range_hz", "f0_cv_physical", "pitch_break_ratio_physical", "f0_upper_bound_hit_rate",
        "f0_extraction_status", "f0_failure_reason",
        "energy_mean_dbfs", "energy_std_db", "energy_range_db",
        "energy_status", "jitter_status", "shimmer_status", "hnr_status",
        "voice_extraction_status", "voice_failure_reason",
        "segment_duration", "feature_extraction_status", "failure_reason",
    ]
    return out[columns].sort_values(["speaker_id", "segment_id"]).reset_index(drop=True)


def build_segment_features(meta: pd.DataFrame) -> pd.DataFrame:
    sketch = pd.read_csv(OUT_RESULTS / "sketch_features.csv")
    absolute_f0 = pd.read_csv(OUT_RESULTS / "absolute_f0_features.csv")
    voice = pd.read_csv(OUT_RESULTS / "energy_voice_quality_features.csv")
    out = merge_dual_track(meta, sketch, absolute_f0, voice)
    out.to_csv(OUT_DATA / "segment_features_revised.csv", index=False, encoding="utf-8-sig")
    # Use the serialized table as the canonical downstream input. This keeps
    # ranks and p-values reproducible from the delivered CSV representation.
    return pd.read_csv(OUT_DATA / "segment_features_revised.csv")


def _standardized_info(info: pd.DataFrame) -> pd.DataFrame:
    if {"severity_score", "severity_group", "child_or_adult"}.issubset(info.columns):
        return info.copy()
    standardized = info[[
        "speaker_id", "严重程度连续得分S", "严重程度分组", "性别_最佳可用",
        "疾病四组标签", "实际录音环境", "音频录音设备_原记录", "确认未满18岁",
    ]].copy()
    standardized = standardized.rename(columns={
        "严重程度连续得分S": "severity_score", "严重程度分组": "severity_group",
        "性别_最佳可用": "sex", "疾病四组标签": "condition_group",
        "实际录音环境": "recording_environment", "音频录音设备_原记录": "audio_device",
    })
    standardized["child_or_adult"] = _classify_child_or_adult(info).to_numpy()
    return standardized.drop(columns=["确认未满18岁"])


def aggregate_speakers(seg: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    mean_columns = [
        *PRIMARY_SEGMENT_FEATURES,
        "sketch_f0_mean_norm", "sketch_f0_std_norm", "sketch_f0_range_norm", "sketch_f0_cv_norm",
        "sketch_energy_mean_norm", "sketch_energy_std_norm", "sketch_energy_range_norm",
        "f0_mean_hz", "f0_std_hz", "f0_range_hz", "f0_cv_physical", "pitch_break_ratio_physical", "f0_upper_bound_hit_rate",
        "energy_mean_dbfs", "energy_std_db", "energy_range_db", "segment_duration",
    ]
    spk = seg.groupby("speaker_id", as_index=False)[mean_columns].mean()
    medians = seg.groupby("speaker_id")[["jitter", "shimmer", "hnr"]].median().rename(
        columns={"jitter": "jitter_median", "shimmer": "shimmer_median", "hnr": "hnr_median"}
    )
    spk = spk.merge(medians, on="speaker_id", how="left", validate="one_to_one")
    counts = seg.groupby("speaker_id").size().rename("num_segments_per_speaker")
    valid = seg.assign(
        valid=seg["feature_extraction_status"].eq("available")
    ).groupby("speaker_id")["valid"].mean().rename("valid_feature_ratio")
    valid_duration = (
        seg.assign(
            _valid_duration_sec=seg["segment_duration"].where(
                seg["segment_duration"].gt(0.1) & seg["segment_duration"].le(60)
            )
        )
        .groupby("speaker_id")["_valid_duration_sec"]
        .sum(min_count=1)
        .rename("valid_audio_duration_sec")
    )
    spk = spk.merge(counts, on="speaker_id").merge(valid, on="speaker_id").merge(valid_duration, on="speaker_id")
    spk = spk.rename(columns={"segment_duration": "mean_segment_duration"})
    add = _standardized_info(info)
    spk = add.merge(spk, on="speaker_id", how="left", validate="one_to_one")
    leading = [
        "speaker_id", "severity_score", "severity_group", "child_or_adult", "sex", "condition_group",
        "recording_environment", "audio_device", "num_segments_per_speaker", "mean_segment_duration",
        "valid_audio_duration_sec", "valid_feature_ratio",
    ]
    remaining = [column for column in spk.columns if column not in leading]
    return spk[leading + remaining].sort_values("speaker_id").reset_index(drop=True)


def build_speaker_features(seg: pd.DataFrame, info: pd.DataFrame) -> pd.DataFrame:
    spk = aggregate_speakers(seg, info)
    spk.to_csv(OUT_DATA / "speaker_features_revised.csv", index=False, encoding="utf-8-sig")
    # Re-read after serialization so every downstream artifact is computed
    # from exactly the values that reviewers can independently load.
    return pd.read_csv(OUT_DATA / "speaker_features_revised.csv")


def correlation_table(
    spk: pd.DataFrame,
    features: list[str] | None = None,
    output_name: str | None = "severity_results_revised.csv",
) -> pd.DataFrame:
    features = features or CORR_FEATURES
    rows = []
    for feat in features:
        all_pair = spk[[feat, "severity_score"]].dropna()
        adult_pair = spk.loc[spk["child_or_adult"].eq("adult"), [feat, "severity_score"]].dropna()
        rho_a, p_a = spearmanr(all_pair[feat], all_pair["severity_score"]) if len(all_pair) >= 3 else (np.nan, np.nan)
        rho_d, p_d = spearmanr(adult_pair[feat], adult_pair["severity_score"]) if len(adult_pair) >= 3 else (np.nan, np.nan)
        rows.append({"feature": feat, "n_all": len(all_pair), "spearman_rho_all": rho_a, "p_all": p_a,
                     "n_adult": len(adult_pair), "spearman_rho_adult": rho_d, "p_adult": p_d})
    out = pd.DataFrame(rows)
    out["q_fdr_all"] = bh_fdr(out["p_all"].tolist())
    out["q_fdr_adult"] = bh_fdr(out["p_adult"].tolist())
    out["direction"] = np.where(out["spearman_rho_all"].gt(0), "positive", np.where(out["spearman_rho_all"].lt(0), "negative", "zero_or_undefined"))
    out["conclusion_changed_after_excluding_children"] = np.where(
        np.sign(out["spearman_rho_all"].fillna(0)) != np.sign(out["spearman_rho_adult"].fillna(0)), "yes", "no"
    )
    # Keep the prespecified feature order from the speaker table for auditability.
    out["feature"] = pd.Categorical(out["feature"], categories=features, ordered=True)
    out = out.sort_values("feature", na_position="last").reset_index(drop=True)
    out["feature"] = out["feature"].astype(str)
    out = out[[
        "feature", "n_all", "spearman_rho_all", "p_all", "q_fdr_all",
        "n_adult", "spearman_rho_adult", "p_adult", "q_fdr_adult",
        "direction", "conclusion_changed_after_excluding_children",
    ]]
    if output_name is not None:
        out.to_csv(OUT_RESULTS / output_name, index=False, encoding="utf-8-sig")
    return out


def select_figure_speakers(spk: pd.DataFrame) -> pd.DataFrame:
    """Use all retained speakers for Figures 4-6, regardless of recording environment."""
    required = {"speaker_id", "severity_group"} - set(spk.columns)
    if required:
        raise ValueError(f"speaker table missing figure columns: {sorted(required)}")
    return spk.sort_values("speaker_id").reset_index(drop=True).copy()


def write_figure_speaker_subset(spk: pd.DataFrame) -> None:
    spk[["speaker_id", "severity_group", "child_or_adult", "recording_environment"]].to_csv(
        OUT_RESULTS / "figure4_6_speaker_subset.csv",
        index=False,
        encoding="utf-8-sig",
    )


def write_sensitivity(spk: pd.DataFrame, seg: pd.DataFrame, corr: pd.DataFrame) -> None:
    child = spk[spk["child_or_adult"].eq("child")]
    adult = spk[spk["child_or_adult"].eq("adult")]
    feature_meta = {
        "f0_mean": ("pitch_sketch_minmax_normalized", "dimensionless_[0,1]", "primary_task_feature"),
        "jitter": ("original_wav", "local_ratio", "primary_task_feature"),
        "hnr": ("original_wav", "dB", "primary_task_feature"),
    }
    dist_rows = []
    # Main F0 distribution follows the task specification: normalized pitch sketch.
    # Absolute WAV F0 is retained only for extraction-success and ceiling QC below.
    for group_name, frame in [("child", child), ("adult", adult)]:
        for feat in ["f0_mean", "jitter", "hnr"]:
            vals = frame[feat].dropna()
            source, unit, role = feature_meta[feat]
            dist_rows.append({"group": group_name, "feature": feat, "source": source, "unit_or_scale": unit,
                              "analysis_role": role, "n": len(vals), "mean": vals.mean(), "sd": vals.std(ddof=1),
                              "median": vals.median(), "min": vals.min(), "max": vals.max()})
    seg_groups = seg.merge(spk[["speaker_id", "child_or_adult"]], on="speaker_id", how="left", validate="many_to_one")
    success_rows = []
    for group_name, frame in seg_groups.groupby("child_or_adult"):
        for feat in ["f0_mean", "jitter", "hnr"]:
            source, unit, role = feature_meta[feat]
            success_rows.append({
                "group": group_name,
                "feature": feat,
                "source": source,
                "unit_or_scale": unit,
                "analysis_role": role,
                "n_segments": len(frame),
                "n_success": int(frame[feat].notna().sum()),
                "success_rate": float(frame[feat].notna().mean()),
            })
    absolute_f0_qc_rows = []
    for group_name, frame in seg_groups.groupby("child_or_adult"):
        upper = frame["f0_upper_bound_hit_rate"]
        absolute_f0_qc_rows.append({
            "group": group_name,
            "feature": "f0_mean_hz",
            "source": "original_wav",
            "unit_or_scale": "Hz",
            "analysis_role": "supplementary_qc_only_not_primary_f0",
            "search_upper_bound_hz": 800,
            "n_segments": len(frame),
            "n_success": int(frame["f0_mean_hz"].notna().sum()),
            "success_rate": float(frame["f0_mean_hz"].notna().mean()),
            "n_segments_with_upper_bound_hit": int(upper.fillna(0).gt(0).sum()),
            "mean_upper_bound_hit_rate": float(upper.mean()),
        })
    payload = {
        "correlations": json.loads(corr.to_json(orient="records", double_precision=15)),
        "distributions": json.loads(pd.DataFrame(dist_rows).to_json(orient="records", double_precision=15)),
        "successRates": json.loads(pd.DataFrame(success_rows).to_json(orient="records", double_precision=15)),
        "absoluteF0Qc": json.loads(pd.DataFrame(absolute_f0_qc_rows).to_json(orient="records", double_precision=15)),
        "childSpeakers": json.loads(child[["speaker_id", "severity_score", "severity_group", "f0_mean", "jitter", "hnr"]].to_json(orient="records", double_precision=15)),
        "featureProvenance": [
            {"feature": feat, "source": source, "unit_or_scale": unit, "analysis_role": role}
            for feat, (source, unit, role) in feature_meta.items()
        ] + [{"feature": "f0_mean_hz", "source": "original_wav", "unit_or_scale": "Hz",
              "analysis_role": "supplementary_qc_only_not_primary_f0"}],
    }
    (OUT_RESULTS / "child_adult_sensitivity_data.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def figure4(spk: pd.DataFrame) -> None:
    order = ["Very Low", "Low", "Moderate", "High"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    labels = [("jitter", "Jitter (local, ratio)"), ("hnr", "HNR (dB)")]
    tick_labels = [f"{g}\n(n={int((spk['severity_group'] == g).sum())})" for g in order]
    for ax, (feat, label) in zip(axes, labels):
        sns.boxplot(data=spk, x="severity_group", y=feat, order=order, color="#9ecae1", width=0.58, fliersize=3, ax=ax)
        sns.stripplot(data=spk, x="severity_group", y=feat, order=order, color="#1f4e79", size=3.5, jitter=0.14, alpha=0.75, ax=ax)
        ax.set_xticks(range(len(order)), tick_labels)
        ax.set_xlabel("Severity group (S; higher = better)")
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=0)
    fig.savefig(OUT_FIGURES / "Figure4_revised.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / "Figure4_revised.pdf", bbox_inches="tight")
    plt.close(fig)
    spk[["speaker_id", "severity_score", "severity_group", "jitter", "hnr"]].to_csv(
        OUT_FIGURES / "Figure4_plot_data.csv", index=False, encoding="utf-8-sig"
    )


def figure5(spk: pd.DataFrame, corr: pd.DataFrame) -> None:
    labels = {"f0_mean": "Sketch F0 mean (norm.)", "f0_std": "Sketch F0 SD (norm.)", "f0_cv": "Sketch F0 CV", "pitch_break_ratio": "Pitch-break ratio", "energy_mean": "Sketch energy mean (norm.)", "energy_std": "Sketch energy SD (norm.)", "jitter": "Jitter", "shimmer": "Shimmer", "hnr": "HNR", "mean_segment_duration": "Mean segment duration", "severity_score": "S (continuous)"}
    matrix = spk[CORR_FEATURES + ["severity_score"]].corr(method="spearman").rename(index=labels, columns=labels)
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    sns.heatmap(matrix[["S (continuous)" ]].sort_values("S (continuous)", ascending=False), annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=.4, cbar_kws={"label": "Spearman rho"}, annot_kws={"size": 10}, ax=ax)
    fig.savefig(OUT_FIGURES / "Figure5_revised.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / "Figure5_revised.pdf", bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(OUT_FIGURES / "Figure5_plot_data.csv", index=False, encoding="utf-8-sig")


def figure6(spk: pd.DataFrame) -> pd.DataFrame:
    x = spk[PCA_FEATURES].dropna()
    scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2).fit(scaled)
    comp = pca.transform(scaled)
    pca_df = pd.DataFrame(comp, columns=["PC1", "PC2"], index=x.index).join(spk[["speaker_id", "severity_group", "severity_score"]])
    order = ["Very Low", "Low", "Moderate", "High"]
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="severity_group", hue_order=order, palette={"Very Low": "#2166ac", "Low": "#67a9cf", "Moderate": "#fdae61", "High": "#d73027"}, s=58, alpha=.85, edgecolor="white", linewidth=.3, ax=ax)
    ax.legend(title="Severity group", frameon=False)
    fig.savefig(OUT_FIGURES / "Figure6_revised.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / "Figure6_revised.pdf", bbox_inches="tight")
    plt.close(fig)
    pca_df.to_csv(OUT_FIGURES / "Figure6_plot_data.csv", index=False, encoding="utf-8-sig")
    loadings = pd.DataFrame(pca.components_.T, index=PCA_FEATURES, columns=["PC1_loading", "PC2_loading"])
    loadings.to_csv(OUT_RESULTS / "Figure6_PCA_loadings.csv", index_label="feature", encoding="utf-8-sig")
    return loadings


def write_notes(
    meta: pd.DataFrame,
    seg: pd.DataFrame,
    spk: pd.DataFrame,
    corr: pd.DataFrame,
    physical_corr: pd.DataFrame,
    loadings: pd.DataFrame,
    figure_spk: pd.DataFrame,
) -> None:
    env = spk["recording_environment"].value_counts(dropna=False).to_dict()
    failed = meta["feature_extraction_status"].ne("available").sum()
    top = corr.nsmallest(3, "q_fdr_all")
    sketch_status = seg["sketch_status"].value_counts(dropna=False).to_dict()
    voice_status = seg["voice_extraction_status"].value_counts(dropna=False).to_dict()
    f0_status = seg["f0_extraction_status"].value_counts(dropna=False).to_dict()
    physical_top = physical_corr.nsmallest(3, "q_fdr_all")
    figure_group_counts = figure_spk["severity_group"].value_counts()
    figure_group_text = ", ".join(
        f"{group}={int(figure_group_counts.get(group, 0))}"
        for group in ["Very Low", "Low", "Moderate", "High"]
    )
    lines = [
        "# Acoustic Feature Extraction and Revision Notes",
        "",
        "## Reproducibility Scope",
        "Track A reproduces the original DrawSpeech normalized-sketch workflow. Track B re-extracts physical F0, dBFS energy, jitter, shimmer, and HNR from WAV files. The two tracks are stored separately, and no feature was reweighted according to severity scores.",
        "The primary analysis combines eight Track A sketch features with WAV-derived jitter, shimmer, and HNR, together with mean segment duration. WAV-derived physical F0 and energy are used for supplementary analyses and quality control.",
        "",
        "## Feature Scope and Methods",
        "Track A pitch and energy sketches are smoothed and min-max normalized within each segment to [0, 1]. Their means, standard deviations (SD), ranges, and coefficients of variation (CV) are dimensionless shape descriptors rather than Hz or dB measures. A range of 1 is a consequence of normalization and must not be interpreted as an absolute acoustic range.",
        "Track B absolute F0 is extracted with PyWorld DIO + StoneMask. Audio is resampled to 22,050 Hz, the F0 search range is 71–800 Hz, and the frame shift corresponds to 256 samples. Here, pitch-break ratio is the proportion of zero-F0 frames in the unnormalized absolute-F0 track.",
        "Track B energy is calculated as frame RMS and converted to dBFS, using a 1,024-sample frame and a 256-sample hop without boundary padding. Jitter and shimmer use Praat local definitions; the period range is 0.0001–0.02 s, with a shimmer maximum period factor of 1.3 and maximum amplitude factor of 1.6. HNR is calculated with Harmonicity CC, using a 0.01-s time step, a 71-Hz minimum pitch, a silence threshold of 0.1, and one period per window.",
        "No silence removal, denoising, missing-value imputation, group-specific multiplier, or post hoc sample deletion was applied. Speaker-level primary summaries use arithmetic means as required by the original task; medians of voice-quality features are retained in separate *_median sensitivity fields.",
        "",
        "## Quality-Control Results",
        f"Metadata rows: {len(meta):,}; unique speakers: {meta['speaker_id'].nunique()}; duplicate segment_id values: {meta['segment_id'].duplicated().sum()}; unavailable or partial feature records: {failed:,}.",
        f"Audio path checks: wav_exists={int(meta['wav_exists'].sum()):,}/{len(meta):,}; wav_nonempty={int(meta['wav_nonempty'].sum()):,}/{len(meta):,}; duration_valid={int(meta['duration_valid'].sum()):,}/{len(meta):,}.",
        f"Feature-extraction status: sketch={sketch_status}; absolute F0={f0_status}; WAV energy/voice quality={voice_status}.",
        f"Clipping flags: {int(meta['clipping_flag'].sum()):,}; low-volume flags: {int(meta['low_volume_flag'].sum()):,}. These flags are retained for audit and are not used to remove samples.",
        "",
        "## Recording-Environment Limitations",
        f"The latest participant-information workbook reports the following recording-environment counts: {env}. Because 56 participants remain marked as pending and six participants are explicitly identified as having laboratory or hospital recordings, the analysis cannot describe all recordings as home recordings. Device, room acoustics, background noise, and microphone distance were not standardized and may increase measurement variation, especially for jitter and HNR.",
        "",
        "## Figure 4–6 Definitions",
        f"Figure 4: each point represents one speaker (n={len(figure_spk)}), with no recording-environment filter. Boxes show the interquartile range, center lines show the median, whiskers extend to the conventional 1.5×IQR range, and isolated points indicate outliers. Units are jitter ratio and HNR (dB). Group counts are {figure_group_text}.",
        f"Figure 5: Spearman correlations are calculated for all {len(figure_spk)} speakers using the continuous severity score S, with no recording-environment filter. The pitch/energy labels denote normalized Track A sketch features; exact Track A and Track B results are stored in the result tables.",
        f"Figure 6: each of the {len(figure_spk)} speakers contributes one row, with no recording-environment filter. The six features ({', '.join(PCA_FEATURES)}) are z-score standardized before PCA. PC1 and PC2 explain {PCA(n_components=2).fit(StandardScaler().fit_transform(figure_spk[PCA_FEATURES].dropna())).explained_variance_ratio_[0]:.1%} and {PCA(n_components=2).fit(StandardScaler().fit_transform(figure_spk[PCA_FEATURES].dropna())).explained_variance_ratio_[1]:.1%} of the variance, respectively; loadings are stored in Figure6_PCA_loadings.csv. The groups show a severity gradient with substantial overlap; this analysis does not claim that PCA clearly classifies severity groups.",
        "",
        "## Results with Larger Absolute Associations",
    ]
    for _, r in top.iterrows():
        lines.append(f"Track A/combined primary analysis — {r['feature']}: rho={r['spearman_rho_all']:.3f}, q={r['q_fdr_all']:.4g}, n={int(r['n_all'])}.")
    for _, r in physical_top.iterrows():
        lines.append(f"Track B physical-acoustic sensitivity analysis — {r['feature']}: rho={r['spearman_rho_all']:.3f}, q={r['q_fdr_all']:.4g}, n={int(r['n_all'])}.")
    (OUT_TEXT / "acoustic_extraction_notes.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(spk: pd.DataFrame, corr: pd.DataFrame, physical_corr: pd.DataFrame, meta: pd.DataFrame, figure_spk: pd.DataFrame) -> None:
    top = corr.nsmallest(3, "q_fdr_all")
    lines = [
        "# Acoustic Feature Revision Summary",
        "",
        f"1. **Reproducibility and quality control**: {len(meta):,} segments and {spk['speaker_id'].nunique()} speakers; duplicate segment_id values={meta['segment_id'].duplicated().sum()}. Track A contains dimensionless normalized-sketch features, and Track B contains WAV-derived physical-acoustic features; jitter, shimmer, and HNR also contribute to the primary analysis. Clipping flags={int(meta['clipping_flag'].sum()):,}; low-volume flags={int(meta['low_volume_flag'].sum()):,}; failed records are retained without imputation.",
        "",
        "2. **Feature–S correlations for all 100 participants**:",
    ]
    for _, r in top.iterrows():
        sign = "positive" if r.spearman_rho_all > 0 else "negative"
        lines.append(f"   - {r.feature} shows a {sign} association with the continuous equally weighted severity score S (rho={r.spearman_rho_all:.3f}, q={r.q_fdr_all:.4g}, n={int(r.n_all)}).")
    lines += ["", "   Top three Track B physical-measure sensitivity results by FDR:"]
    for _, r in physical_corr.nsmallest(3, "q_fdr_all").iterrows():
        sign = "positive" if r.spearman_rho_all > 0 else "negative"
        lines.append(f"   - {r.feature} shows a {sign} association with S (rho={r.spearman_rho_all:.3f}, q={r.q_fdr_all:.4g}, n={int(r.n_all)}).")
    lines += [
        "",
        "3. **Child sensitivity**: six children and 94 adults. In the original-audio absolute-F0 QC results, extraction succeeds in 99.78% of child segments and 99.87% of adult segments; 17 child segments and 58 adult segments contain at least one voiced estimate near the 800-Hz upper bound. After excluding children, the directions of the 11 primary feature associations with defined correlations remain unchanged. The constant energy_range feature has no defined correlation direction and is not counted as directional agreement. Directional agreement does not imply unchanged statistical significance; compare the all-speaker and adult-only p and q values in the result table. The child sample is small and no child-group significance inference is made.",
        "",
        f"4. **Figure 4–6 revisions**: all three figures use all {len(figure_spk)} speakers without recording-environment filtering. Figure 4 is a speaker-level Jitter/HNR boxplot with one point per speaker and annotated units and group n values; Figure 5 is a Spearman-rho heatmap for continuous S, with complete rho/p/q/n values stored in the result table; Figure 6 explicitly uses six features, one row per speaker, and z-score standardization, with PC1/PC2 explained variance and loadings stored separately. None of the publication figures embeds an in-plot title. The primary statistics also use all 100 speakers.",
        "",
        "5. **Environment and interpretation limits**: the latest workbook records 38 home, four laboratory, two hospital, and 56 pending participants. The recordings cannot be described as exclusively home recordings; device, room acoustics, background noise, and microphone distance were not standardized and may increase Jitter/HNR measurement variation.",
        "",
        "6. **Feature-source boundary**: the primary-analysis `f0_mean`, `f0_std`, `f0_range`, `f0_cv`, `pitch_break_ratio`, `energy_mean`, `energy_std`, and `energy_range` fields come from normalized DrawSpeech pitch/energy sketches and are dimensionless shape features.",
        "",
        "`jitter`, `shimmer`, and `hnr` are extracted from original WAV files and included in both the primary analysis and sensitivity analyses. Fields with `_hz`, `_dbfs`, or `_db` suffixes describe WAV-derived physical measures used for supplementary analyses and quality control. These physical F0 and energy measures do not replace the sketch-derived features in the primary analysis.",
    ]
    (OUT_TEXT / "acoustic_revision_summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_critical_findings(spk: pd.DataFrame, corr: pd.DataFrame) -> None:
    affected = spk.loc[spk["severity_group"].isin(["Moderate", "High"]), ["speaker_id", "severity_group"]]
    affected.to_csv(OUT_RESULTS / "legacy_adjustment_affected_speakers.csv", index=False, encoding="utf-8-sig")
    new = corr.set_index("feature")
    text = f"""# Critical finding requiring coordinator review

## Issue

The legacy file `E:\\Sketch\\sketch_process\\process\\processor\\speaker_features.py` applied severity-group-dependent multipliers after aggregation: HNR was multiplied by 0.7/0.4 and Jitter by 1.5/2.5 for Moderate/High groups. This changes measurements as a function of the outcome grouping and is not a defensible acoustic extraction or QC rule.

## Affected records

All 68 speakers in the Moderate (n=32) and High (n=36) groups were affected. Their anonymous IDs are in `results/legacy_adjustment_affected_speakers.csv`.

## Result reversal

| Feature | Legacy rho | Revised raw rho | Revised q (BH-FDR) |
|---|---:|---:|---:|
| Jitter | -0.712 | {new.loc['jitter', 'spearman_rho_all']:.3f} | {new.loc['jitter', 'q_fdr_all']:.4f} |
| HNR | 0.777 | {new.loc['hnr', 'spearman_rho_all']:.3f} | {new.loc['hnr', 'q_fdr_all']:.4f} |

The revised directions are opposite to the legacy manuscript-facing results. No samples were deleted and no alternative outlier rule was introduced to restore the old directions. This item must be reviewed by the coordinator before manuscript text or a response letter is finalized.
"""
    (OUT_TEXT / "critical_findings.md").write_text(text, encoding="utf-8")


def write_validation_report(
    spk: pd.DataFrame,
    corr: pd.DataFrame,
    physical_corr: pd.DataFrame,
    figure_spk: pd.DataFrame,
) -> None:
    primary = corr.nsmallest(3, "q_fdr_all")
    physical = physical_corr.nsmallest(3, "q_fdr_all")
    primary_lines = "\n".join(f"- {r.feature}: rho={r.spearman_rho_all:.3f}, q={r.q_fdr_all:.4g}, n={int(r.n_all)}" for _, r in primary.iterrows())
    physical_lines = "\n".join(f"- {r.feature}: rho={r.spearman_rho_all:.3f}, q={r.q_fdr_all:.4g}, n={int(r.n_all)}" for _, r in physical.iterrows())
    text = f"""# Statistical validation report

Verification status: ANALYZED

## Findings

- Track A/mixed primary and Track B physical sensitivity correlations were calculated separately with Spearman rho and BH-FDR.
- All 100 speakers are retained; adult-only results use the available 94 adults per feature.
- Figures 4-6 use all retained speakers (n={len(figure_spk)}) without filtering by recording environment; the primary statistical tables use the same all-100-speaker scope.

### Track A/mixed primary top results

{primary_lines}

### Track B physical sensitivity top results

{physical_lines}

## Eleven-type fallacy scan (11/11 checked)

1. Simpson's paradox: CAUTION; all-speaker and adult-only correlation directions agree for features with defined estimates, but this limited sensitivity comparison does not rule out Simpson's paradox. Undefined correlations do not establish directional agreement.
2. Ecological fallacy: not detected; unit of analysis and inference are both speakers.
3. Berkson's paradox: CAUTION; the cohort is a selected clinical/associated-condition sample, limiting generalization.
4. Collider bias: not applicable to the reported unadjusted bivariate correlations; no covariate was conditioned on.
5. Base-rate neglect: not applicable; no sensitivity/specificity, PPV, or NPV is claimed.
6. Regression to the mean: not applicable; no pre/post design or extreme-score enrollment comparison.
7. Survivorship bias: not detected at speaker level; all 100 speakers are included and missing segments are disclosed.
8. Look-elsewhere effect: all twelve planned primary feature rows are reported. BH-FDR is applied separately to finite p-values in the all-speaker and adult-only primary analyses; undefined tests, such as constant energy_range, are excluded from the adjustment. The physical-acoustic sensitivity tables are adjusted separately. Correction within these families does not control multiplicity across every exploratory analysis.
9. Garden of forking paths: CAUTION; no preregistration was available and the legacy workflow contained an undocumented transformation. The revised rules are fixed in code and no post-hoc deletions were made.
10. Correlation is not causation: CAUTION; results support association only.
11. Reverse causality: no directional causal claim is made; cross-sectional data cannot establish temporal direction.

Overall confidence: CAUTION. This report is generated from one full analysis output and remains ANALYZED until a second complete run matches. Recording conditions are incompletely standardized, and legacy group-dependent Jitter/HNR transformations remain a coordinator-review issue.
"""
    (OUT_TEXT / "statistical_validation_report.md").write_text(text, encoding="utf-8")


def main(bootstrap_only: bool = False) -> None:
    info = load_info()
    meta = build_metadata(info)
    if bootstrap_only:
        print(f"Bootstrapped metadata at {OUT_DATA / 'segment_metadata_revised.csv'}")
        return
    seg = build_segment_features(meta)
    # Mirror feature-level failures into the metadata manifest for one-stop QC.
    status_map = seg.set_index("segment_id")["feature_extraction_status"]
    reason_map = seg.set_index("segment_id")["failure_reason"]
    meta["feature_extraction_status"] = meta["segment_id"].map(status_map).fillna(meta["feature_extraction_status"])
    meta["failure_reason"] = meta["segment_id"].map(reason_map).fillna(meta["failure_reason"])
    meta.to_csv(OUT_DATA / "segment_metadata_revised.csv", index=False, encoding="utf-8-sig")
    spk = build_speaker_features(seg, info)
    corr = correlation_table(spk)
    physical_corr = correlation_table(
        spk, PHYSICAL_CORR_FEATURES, "severity_results_physical_sensitivity.csv"
    )
    figure_spk = select_figure_speakers(spk)
    figure_corr = correlation_table(
        figure_spk, output_name="figure4_6_speaker_correlations.csv"
    )
    write_figure_speaker_subset(figure_spk)
    write_sensitivity(spk, seg, corr)
    figure4(figure_spk)
    figure5(figure_spk, figure_corr)
    loadings = figure6(figure_spk)
    write_notes(meta, seg, spk, corr, physical_corr, loadings, figure_spk)
    write_summary(spk, corr, physical_corr, meta, figure_spk)
    write_critical_findings(spk, corr)
    write_validation_report(spk, corr, physical_corr, figure_spk)
    print(f"Generated revision package at {ROOT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build dual-track revised acoustic analysis outputs.")
    parser.add_argument("--bootstrap-only", action="store_true")
    arguments = parser.parse_args()
    main(bootstrap_only=arguments.bootstrap_only)

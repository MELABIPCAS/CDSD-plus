# Feature Provenance and Use

| Field | Source | Unit / scale | Use |
|---|---|---|---|
| `f0_mean`, `f0_std`, `f0_range`, `f0_cv` | Normalized pitch sketch | Dimensionless | Primary analysis |
| `pitch_break_ratio` | Zero-valued frames in the pitch sketch | Proportion | Primary analysis |
| `energy_mean`, `energy_std`, `energy_range` | Normalized energy sketch | Dimensionless | Primary analysis |
| `jitter`, `shimmer`, `hnr` | Original WAV | Praat physical-acoustic measures | Supplementary table / sensitivity analysis |
| `*_hz`, `*_dbfs`, `*_db` | Original WAV | Hz or dB | Physical-acoustic QC / supplementary track |
| `pitch_break_ratio_physical` | Absolute F0 from the original WAV | Proportion | Physical-F0 QC |

Each sketch is min-max normalized within a segment during construction. Therefore, `f0_range` and `energy_range` are shape statistics. A sketch that spans the full normalized range has a range close to 1; blank or degenerate segments may have a range of 0, and other segments may have values below 1. When `energy_range` behaves as a structural constant in this dataset, its correlation results should not be over-interpreted.

Severity scores and severity groups are not used to multiply or otherwise transform acoustic features. All primary statistics use speaker as the independent unit. Figures 4–6 include all 100 speakers and do not filter by recording environment.

## Legacy normalized-range flags

For successfully extracted sketches, `sketch_f0_range_degenerate` and
`sketch_energy_range_degenerate` are legacy flags indicating whether the
corresponding normalized range is close to 1, as evaluated by
`numpy.isclose(range, 1.0, atol=1e-8)`.

These flags do not identify flat sketches, blank audio, or extraction failures.
For example, a sketch `[0, 0.5, 1]` has range 1 and a true flag, whereas an
all-zero sketch has range 0 and a false flag. When extraction fails, the flags
retain their default false values; always interpret them together with
`sketch_status`.

A range close to 1 is expected after segment-wise min-max normalization.
Whether the range is constant across speakers must be assessed from the
aggregated speaker-level values; these segment-level flags alone do not
establish that conclusion.

The legacy names and calculations are retained for compatibility with the
experimental outputs. These flags are not used to exclude segments or
speakers from the analysis.

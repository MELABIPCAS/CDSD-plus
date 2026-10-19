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

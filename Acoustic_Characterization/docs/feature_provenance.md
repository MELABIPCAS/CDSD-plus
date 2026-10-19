# 特征来源与用途

| 字段 | 来源 | 单位/尺度 | 用途 |
|---|---|---|---|
| `f0_mean`, `f0_std`, `f0_range`, `f0_cv` | 归一化 pitch sketch | 无量纲 | 主分析 |
| `pitch_break_ratio` | pitch sketch 中零值帧 | 比例 | 主分析 |
| `energy_mean`, `energy_std`, `energy_range` | 归一化 energy sketch | 无量纲 | 主分析 |
| `jitter`, `shimmer`, `hnr` | 原始 WAV | Praat 物理声学量 | 主表补充/敏感性分析 |
| `*_hz`, `*_dbfs`, `*_db` | 原始 WAV | Hz 或 dB | 物理声学 QC/补充轨道 |
| `pitch_break_ratio_physical` | 原始 WAV 绝对 F0 | 比例 | 物理 F0 QC |

草图在构建时已经逐片段 min-max 归一化，因此 `f0_range` 和 `energy_range` 是形状统计量。若草图覆盖完整归一化范围，范围接近 1；空白或退化片段可能为 0，其他片段也可能小于 1。`energy_range` 在本批数据中表现为结构性常数时，不对其相关结果做过度解释。

严重程度得分或严重程度分组不参与任何声学特征的乘数变换。所有主要统计以 speaker 为独立单位；Figure 4–6 使用全部 100 名 speaker，不按录音环境筛选。

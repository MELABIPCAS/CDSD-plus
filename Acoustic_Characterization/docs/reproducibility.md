# 复现说明

## 输入

运行前需要准备三类私有输入：工作区中的 `segment_metadata.csv`，DrawSpeech 数据根目录下的 WAV 和 pitch/energy sketch，以及最新被试信息 Excel。路径通过 `config/paths.local.json` 或环境变量提供，不把输入复制进代码仓库。

## 阶段

`run_full_reanalysis.py` 依次执行：

1. `run_acoustic_revision.py --bootstrap-only`：读取元数据和被试信息，生成片段级入口表；
2. `run_audio_qc.py`：检查 WAV 可读性、采样率、声道、削波和音量；
3. `run_absolute_f0.py`：从原始 WAV 提取绝对 F0 及搜索上限命中 QC；
4. `run_sketch_features.py`：从已有归一化草图提取 Track A 片段特征；
5. `run_energy_voice_quality.py`：从原始 WAV 提取物理能量、Jitter、Shimmer、HNR；
6. `run_acoustic_revision.py`：合并双轨特征，聚合 speaker，计算相关、敏感性分析和 Figure 4–6。

所有输出都位于 `CDSD_OUTPUT_ROOT` 下的 `data/`、`results/`、`figures/` 和 `text/`。脚本会生成运行日志和 `results/run_manifest.json`，这些文件用于本地审计，不应上传私有数据。

## 验证

```powershell
python scripts/run_full_reanalysis.py --dry-run
python -m compileall scripts
pytest -q
```

正式结果应与独立保存的最终提交包逐项核对；本仓库只负责保存生成逻辑，不替代最终结果包。

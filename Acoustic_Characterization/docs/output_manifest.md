# 输出清单

数据依赖运行产生以下本地输出：

- `data/segment_metadata_revised.csv`：片段元数据、路径、状态和音频 QC；
- `data/segment_features_revised.csv`：双轨片段特征；
- `data/speaker_features_revised.csv`：speaker 级聚合特征；
- `results/severity_results_revised.csv`：主要 Spearman/FDR 结果；
- `results/child_adult_sensitivity_data.json` 与 `child_adult_sensitivity.xlsx`：儿童/成人敏感性分析；
- `results/audio_quality_control.csv`：音频质量控制；
- `figures/Figure4_revised.*`、`Figure5_revised.*`、`Figure6_revised.*`：图及逐点绘图数据；
- `text/`：脚本生成的方法和结果说明。

这些文件可能包含研究数据或派生数据，默认由 `.gitignore` 排除。正式交付使用独立的 `final_report/B_acoustic_revision` 目录；GitHub 仓库只上传代码、配置示例和文档。

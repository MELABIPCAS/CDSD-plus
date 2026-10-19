# CDSD 声学特征分析代码

本目录是 CDSD 声学特征审稿修改分析的代码仓库整理版。仓库只保存可复现代码、配置示例和方法文档；原始 WAV、DrawSpeech 草图 JSON、被试信息表以及生成的 CSV/XLSX/PDF/PNG 结果不应上传到 GitHub。已经核验的结果文件仍保存在工作区的 `final_report/B_acoustic_revision` 中。

## 目录

```text
github_acoustic_analysis/
├─ README.md
├─ requirements.txt
├─ package.json
├─ .gitignore
├─ config/
│  └─ paths.example.json
├─ scripts/
│  ├─ project_config.py
│  ├─ run_full_reanalysis.py
│  ├─ run_acoustic_revision.py
│  ├─ batch_extract_sketches.py
│  ├─ run_audio_qc.py
│  ├─ run_absolute_f0.py
│  ├─ run_sketch_features.py
│  ├─ run_energy_voice_quality.py
│  ├─ acoustic_feature_utils.py
│  └─ build_sensitivity_workbook.mjs
├─ docs/
├─ data/.gitkeep
├─ results/.gitkeep
├─ figures/.gitkeep
└─ tests/
```

## 安装

使用 Python 3.10 或更高版本建立虚拟环境，然后安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需生成儿童/成人敏感性工作簿，再安装 Node.js 18 或更高版本并执行 `npm install`。

## 配置私有数据路径

复制 `config/paths.example.json` 为 `config/paths.local.json`，将四个路径改为本机路径。`paths.local.json` 已被 `.gitignore` 排除。也可以直接设置同名环境变量：

```powershell
$env:CDSD_WORKSPACE_DATA = 'D:\path\to\CDSD_Additional_Analysis\data'
$env:CDSD_PROJECT_DATA = 'E:\path\to\sketch_process\data'
$env:CDSD_INFO_XLSX = 'D:\path\to\CDSD_Plus_100名被试最新信息总表.xlsx'
$env:CDSD_OUTPUT_ROOT = 'D:\path\to\acoustic-analysis-output'
```

其中：

- `CDSD_WORKSPACE_DATA`：包含原始 `segment_metadata.csv` 的工作区数据目录；
- `CDSD_PROJECT_DATA`：包含 `wav` 和已生成草图目录的 DrawSpeech 数据根目录；
- `CDSD_INFO_XLSX`：最新被试信息 Excel；
- `CDSD_OUTPUT_ROOT`：生成 `data/`、`results/`、`figures/` 和 `text/` 的输出根目录。

不设置 `CDSD_OUTPUT_ROOT` 时，输出会写入本仓库目录，但生成文件会被 `.gitignore` 排除。

## 运行顺序

先确认路径配置，再做不读取数据的命令检查：

```powershell
python scripts/run_full_reanalysis.py --dry-run
python -m compileall scripts
pytest -q
```

完整重算按以下顺序执行：

```powershell
python scripts/run_full_reanalysis.py
```

也可以从指定阶段继续：

```powershell
python scripts/run_full_reanalysis.py --from-stage sketch
```

若需要从 WAV 重新构建 DrawSpeech 草图，先将 `CDSD_PROJECT_DATA` 配置为包含 `wav/` 的数据根目录，再运行原始批处理脚本：

```powershell
python scripts/batch_extract_sketches.py
```

儿童/成人敏感性 JSON 已生成后，使用 Node 工作簿脚本：

```powershell
npm run build-workbook
```

## 分析口径

主分析保留两条轨道：

- Track A 的 `f0_mean`、`f0_std`、`f0_range`、`f0_cv`、`pitch_break_ratio`、`energy_mean`、`energy_std`、`energy_range` 来自已经 min-max 归一化的 DrawSpeech pitch/energy sketch；这些是无量纲形状特征。
- Track B 的原始 WAV 物理 F0、能量、`jitter`、`shimmer` 和 `hnr` 用于物理声学补充、QC 或敏感性分析，不能替换主分析中的草图 F0。

主要统计、Figure 4–6、PCA 和儿童/成人敏感性分析以 speaker 为独立单位，使用全部 100 名 speaker；录音环境仅作为描述性字段保留。PCA 的六个输入特征为 `f0_mean`、`f0_std`、`f0_cv`、`pitch_break_ratio`、`energy_mean` 和 `energy_std`，分析前做 z-score 标准化。

## 隐私与上传检查

上传前必须确认：

1. `git status` 中没有 WAV、草图 JSON、Excel、CSV、XLSX、PDF、PNG 或日志；
2. `config/paths.local.json` 没有被跟踪；
3. 脚本中没有本机绝对路径、姓名、联系方式或精确年龄；
4. 代码仓库与正式结果包分开，正式结果按任务书要求另行交付。

特征定义、来源和输出说明见 `docs/feature_provenance.md`；复现步骤见 `docs/reproducibility.md`。

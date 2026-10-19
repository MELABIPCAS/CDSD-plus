# CDSD Acoustic Characterization

This repository contains the reproducible code, configuration example, and method documentation for the CDSD acoustic-feature revision analysis. Private WAV files, DrawSpeech sketch JSON files, participant spreadsheets, and generated CSV/XLSX/PDF/PNG results are intentionally excluded from GitHub. The validated delivery package remains under `final_report/B_acoustic_revision` in the local workspace.

## Repository structure

```text
Acoustic_Characterization/
├─ config/
│  └─ paths.example.json
├─ docs/
│  ├─ feature_provenance.md
│  ├─ output_manifest.md
│  ├─ reproducibility.md
│  └─ sketch_generation_provenance.md
├─ scripts/
│  ├─ __init__.py
│  ├─ acoustic_feature_utils.py
│  ├─ batch_extract_sketches.py
│  ├─ build_sensitivity_workbook.mjs
│  ├─ project_config.py
│  ├─ run_absolute_f0.py
│  ├─ run_acoustic_revision.py
│  ├─ run_audio_qc.py
│  ├─ run_energy_voice_quality.py
│  ├─ run_full_reanalysis.py
│  └─ run_sketch_features.py
├─ tests/
│  └─ test_project_config.py
├─ README.md
├─ package.json
└─ requirements.txt
```

`scripts/batch_extract_sketches.py` is the adapted path-configured copy of the original DrawSpeech sketch-generation script. The algorithm and output schema are documented in `docs/sketch_generation_provenance.md`.

## Installation

Create a Python 3.10 or newer virtual environment and install the Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To build the child/adult sensitivity workbook, install Node.js 18 or newer and run `npm install`.

## Configure private data paths

Copy `config/paths.example.json` to `config/paths.local.json` and replace the example paths with local paths. `paths.local.json` is ignored by Git. The same values may be provided through environment variables:

```powershell
$env:CDSD_WORKSPACE_DATA = 'D:\path\to\CDSD_Additional_Analysis\data'
$env:CDSD_PROJECT_DATA = 'E:\path\to\sketch_process\data'
$env:CDSD_INFO_XLSX = 'D:\path\to\participant_information.xlsx'
$env:CDSD_OUTPUT_ROOT = 'D:\path\to\acoustic-analysis-output'
```

The variables mean:

- `CDSD_WORKSPACE_DATA`: workspace data directory containing the input `segment_metadata.csv`;
- `CDSD_PROJECT_DATA`: DrawSpeech data root containing `wav/` and the generated sketch directory;
- `CDSD_INFO_XLSX`: the latest participant-information workbook;
- `CDSD_OUTPUT_ROOT`: output root for `data/`, `results/`, `figures/`, and `text/`.

If `CDSD_OUTPUT_ROOT` is not set, generated files are written under the repository, where they remain ignored by `.gitignore`.

## Run the analysis

Check the configuration without reading private data:

```powershell
python scripts/run_full_reanalysis.py --dry-run
python -m compileall scripts
pytest -q
```

Run the complete reanalysis in the prescribed order:

```powershell
python scripts/run_full_reanalysis.py
```

To resume from a specific stage:

```powershell
python scripts/run_full_reanalysis.py --from-stage sketch
```

To rebuild DrawSpeech sketches from WAV files, configure `CDSD_PROJECT_DATA` as the data root containing `wav/`, then run:

```powershell
python scripts/batch_extract_sketches.py
```

After the child/adult sensitivity JSON has been generated, build the workbook with:

```powershell
npm run build-workbook
```

## Analysis conventions

The primary analysis keeps two feature tracks:

- Track A: `f0_mean`, `f0_std`, `f0_range`, `f0_cv`, `pitch_break_ratio`, `energy_mean`, `energy_std`, and `energy_range` are extracted from the min-max-normalized DrawSpeech pitch/energy sketches. They are dimensionless shape descriptors.
- Track B: physical F0, energy, `jitter`, `shimmer`, and `hnr` are extracted from the original WAV files for physical-acoustic supplementary analyses, quality control, or sensitivity analyses. They do not replace the sketch-based F0 features in the primary analysis.

The main statistics, Figures 4–6, PCA, and child/adult sensitivity analyses use speaker as the independent unit and include all 100 speakers. Recording environment is retained as a descriptive field. PCA uses six z-score-standardized inputs: `f0_mean`, `f0_std`, `f0_cv`, `pitch_break_ratio`, `energy_mean`, and `energy_std`.

## Upload checklist

Before pushing to GitHub, confirm that:

1. `git status` contains no WAV, sketch JSON, Excel, CSV, XLSX, PDF, PNG, or log files;
2. `config/paths.local.json` is not tracked;
3. scripts contain no machine-specific absolute paths, names, contact details, or exact ages;
4. the code repository remains separate from the formal results package.

Feature definitions and source tracks are described in `docs/feature_provenance.md`; the reproducibility workflow is described in `docs/reproducibility.md`.

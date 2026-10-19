# Reproducibility

## Inputs

Before running the pipeline, prepare three private input sources: the workspace `segment_metadata.csv`, the WAV files and pitch/energy sketches under the DrawSpeech data root, and the latest participant-information workbook. Supply paths through `config/paths.local.json` or environment variables; do not copy private inputs into the repository.

## Stages

`run_full_reanalysis.py` executes the following stages in order:

1. `run_acoustic_revision.py --bootstrap-only`: read metadata and participant information and create the segment-level entry table;
2. `run_audio_qc.py`: check WAV readability, sampling rate, channel count, clipping, and volume;
3. `run_absolute_f0.py`: extract absolute F0 from the original WAV and record upper-bound-hit QC;
4. `run_sketch_features.py`: extract Track A segment features from existing normalized sketches;
5. `run_energy_voice_quality.py`: extract physical energy, Jitter, Shimmer, and HNR from the original WAV;
6. `run_acoustic_revision.py`: merge the two tracks, aggregate speaker-level features, and calculate correlations, sensitivity analyses, and Figures 4–6.

All outputs are written under `CDSD_OUTPUT_ROOT` in `data/`, `results/`, `figures/`, and `text/`. The scripts also create run logs and `results/run_manifest.json` for local auditing; these files should not be uploaded with private data.

## Verification

```powershell
python scripts/run_full_reanalysis.py --dry-run
python -m compileall scripts
pytest -q
```

Compare formal results against the independently stored delivery package item by item. This repository stores the generation logic and does not replace the formal results package.

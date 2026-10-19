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

## Environment-specific XLSX export

The Python pipeline writes `results/child_adult_sensitivity_data.json`. It does not invoke `scripts/build_sensitivity_workbook.mjs` or generate the XLSX workbook.

The original workbook exporter imports `@oai/artifact-tool`, a private package supplied by a specialized execution environment. The public npm registry does not provide this package. The optional peer declaration in `package.json` documents this external runtime requirement; it does not install or pin the package.

Only when the package and a compatible Node.js runtime are already provisioned and package resolution succeeds from this project can the workbook be exported with:

```powershell
npm run build-workbook
```

This command reads the sensitivity JSON and writes `results/child_adult_sensitivity.xlsx` and sheet previews under `results/workbook_preview/`. Without the required runtime, XLSX export is unavailable; the JSON output remains available for inspection. The historical workbook-package version was not recorded, so exact reproduction of its execution environment and workbook rendering is not established. The exporter source is preserved without algorithm changes. Replacing it with a public spreadsheet library would be a separate implementation requiring output validation.

## Verification

```powershell
python scripts/run_full_reanalysis.py --dry-run
python -m compileall scripts
pytest -q
```

Compare formal results against the independently stored delivery package item by item. This repository stores the generation logic and does not replace the formal results package.

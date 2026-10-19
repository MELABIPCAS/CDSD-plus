# Output Manifest

Running the analysis with private inputs produces the following local outputs:

- `data/segment_metadata_revised.csv`: segment metadata, paths, extraction status, and audio QC;
- `data/segment_features_revised.csv`: two-track segment-level features;
- `data/speaker_features_revised.csv`: speaker-level aggregated features;
- `results/severity_results_revised.csv`: primary Spearman/FDR results;
- `results/child_adult_sensitivity_data.json` and `child_adult_sensitivity.xlsx`: child/adult sensitivity analysis;
- `results/audio_quality_control.csv`: audio quality-control results;
- `figures/Figure4_revised.*`, `Figure5_revised.*`, and `Figure6_revised.*`: figures and point-level plotting data;
- `text/`: script-generated methods and results notes.

These outputs may contain research data or derived data and are excluded by default through `.gitignore`. The formal delivery package is maintained separately under `final_report/B_acoustic_revision`; this GitHub repository contains only code, configuration examples, and documentation.

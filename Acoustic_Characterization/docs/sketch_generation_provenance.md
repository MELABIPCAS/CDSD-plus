# Sketch-Generation Script Provenance

The repository file `scripts/batch_extract_sketches.py` originates from:

```text
E:\Sketch\sketch_process\sketch\batch_extract_sketches.py
```

The script recursively scans WAV files and writes JSON files containing `pitch_sketch` and `energy_sketch`. The repository version preserves the original feature-extraction algorithm, parameters, Savitzky–Golay smoothing, min-max normalization, checkpoint skipping, multiprocessing, and JSON field structure.

## Code-consistency check

The original script has the following SHA-256 digest:

```text
B51C85866114FB1FD66ABCE45E2591C1DBE1D2290E8058E867F9592BF3B8BD0B
```

During organization, the hard-coded input and output directories were replaced with paths read from `scripts/project_config.py` through `CDSD_PROJECT_DATA`, followed by `wav/` and `processed_sketches_new/`. Human-facing comments and console messages were translated into English. The extraction operations, parameters, control flow, and JSON-writing logic are unchanged; the path configuration and language-only edits do not alter the generated sketch values or JSON schema.

Thus, when run on the same inputs, the repository version uses the same generation process and JSON schema as the experimental source script; only the input and output locations are configuration-driven.

## Prerequisites

Set `CDSD_PROJECT_DATA` to a data root containing:

```text
<CDSD_PROJECT_DATA>\\wav
<CDSD_PROJECT_DATA>\\processed_sketches_new
```

Run:

```powershell
python scripts/batch_extract_sketches.py
```

Original WAV files and generated sketch JSON files are private data and are not committed to the code repository.

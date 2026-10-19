# CDSD-Plus

CDSD-Plus is a Mandarin speech database for speakers with persistent articulation or intelligibility difficulties. It contains **100 unique speakers** and approximately **190 hours of audio**. The first 44 speakers come from CDSD; their earlier recordings and transcripts are unchanged. Nine of the 100 speakers each contributed about ten additional hours. Approximately nine hours have synchronized video, already counted in the audio total.

**Database page (planned):** [http://melab.psych.ac.cn/en/db/cdsd_plus.html](http://melab.psych.ac.cn/en/db/cdsd_plus.html). The page is not yet published.

**Data access:** Applications will open after the database paper is accepted. Applications and downloads are not available now. The planned release is de-identified and controlled-access; video sharing depends on explicit audiovisual consent.

## Acoustic features

The reported analysis extracts features per speech segment and averages valid values per speaker. Local Jitter and harmonic-to-noise ratio (HNR) were measured with Praat through Parselmouth on complete segments, without external voiced-only cropping, silence removal, or denoising. Jitter is a **ratio** and HNR is in **dB**. Separate smoothed pitch and energy contours were min–max normalized within each segment; their descriptors are dimensionless and are not absolute pitch or intensity. The field `pitch_break_ratio` means the proportion of zeros in the normalized pitch contour, not a physiological voice-break estimate. Failed or nonfinite values remain missing.

Five experts rated intelligibility, articulatory regularity, fluency, and stability from 0 to 10; their four scores were summed and then averaged across experts to give continuous performance score `S` (0–40, higher is better). The reported feature analysis uses **speaker-level Spearman correlations with continuous `S`**, two-sided `p` values, and Benjamini–Hochberg `q` values over 11 estimable tests. Constant normalized energy-contour range has an undefined correlation and is excluded from correction. Severity categories are derived from `S` for corpus stratification, not used in place of the continuous-score analysis.

## DSR benchmark

| Protocol | Fixed partition |
| --- | --- |
| `1h_seen-speaker` | Each of 100 speakers' utterances split 8:1:1 into train/development/test. |
| `1h_unseen-speaker` | Whole speakers split 67/10/23; test speakers have no training recordings. |
| `10h_personalization` | The nine speakers' additional long-duration recordings split within speaker 8:1:1. |

The manuscript reports Paraformer, Conformer, and Whisper-large-v2 baselines. The main recognition metric is **character error rate (CER)** on held-out transcripts. The separate speaker-count-matched CP–HI experiment reports aggregate **WER and SER**; those numbers must not be relabeled CER. For target-speaker adaptation, adaptation utterances are excluded from evaluation and the no-adaptation baseline uses the same remaining speech.

## Contributors

- **Acoustic feature analysis:** [Jiaqing Suo](https://github.com/Grandecals).
- **DSR work:** Keyi Feng  and [Xinran Zhao](https://github.com/zxree).

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import editdistance
from collections import defaultdict

def compute_cer_by_speaker(ref_file, hyp_file, output_file):
    """Compute CER (Character Error Rate) for each speaker"""

    # Read reference and hypothesis files
    refs = {}
    hyps = {}

    with open(ref_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                refs[utt_id] = text

    with open(hyp_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt_id, text = parts
                hyps[utt_id] = text

    # Group by speaker
    speaker_data = defaultdict(list)

    for utt_id in refs:
        if utt_id in hyps:
            # Extract speaker ID (assuming format: speaker_utterance)
            speaker_id = utt_id.split('_')[0] if '_' in utt_id else utt_id
            ref_text = refs[utt_id]
            hyp_text = hyps[utt_id]
            speaker_data[speaker_id].append((ref_text, hyp_text))

    # Compute CER for each speaker
    results = []
    for speaker_id, utterances in speaker_data.items():
        total_chars = 0
        total_errors = 0

        for ref_text, hyp_text in utterances:
            # Compute character-level edit distance
            distance = editdistance.eval(ref_text, hyp_text)
            total_errors += distance
            total_chars += len(ref_text)

        if total_chars > 0:
            cer = (total_errors / total_chars) * 100
            results.append((speaker_id, len(utterances), cer))

    # Sort by CER (highest first)
    results.sort(key=lambda x: x[2], reverse=True)

    # Write results
    with open(output_file, 'w', encoding='utf-8') as f:
        for speaker_id, num_utterances, cer in results:
            f.write(f"{speaker_id}\t\t{num_utterances}\t\t{cer:.2f}%\n")

    print(f"CER by speaker computed for {len(results)} speakers")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compute_cer_by_speaker.py <ref_file> <hyp_file> <output_file>")
        sys.exit(1)

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]
    output_file = sys.argv[3]

    compute_cer_by_speaker(ref_file, hyp_file, output_file)

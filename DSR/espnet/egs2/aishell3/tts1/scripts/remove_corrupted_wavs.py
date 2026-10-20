#!/usr/bin/env python3

"""
Filter out corrupted audio entries from a Kaldi-style wav.scp file.

Usage:
    python remove_corrupted_wavs.py \
        --scp /abs/path/to/wav.scp \
        --output /abs/path/to/wav.clean.scp \
        --root /abs/path/to/egs2/aishell/asr1

By default the output file path is derived from the input path by appending
``.clean`` before the file suffix. A companion ``--report`` file may be passed
to save the list of removed uttids and the associated error messages.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

from kaldiio import load_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scp",
        type=Path,
        required=True,
        help="Absolute path to the input wav.scp file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Absolute path to write the filtered wav.scp. "
        "Defaults to <scp>.clean if not provided.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help=(
            "Root directory used to resolve relative archive paths. "
            "Defaults to the parent directory of the input wav.scp."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write the list of removed utterances and errors.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress progress logs.",
    )
    return parser.parse_args()


def _resolve_spec(spec: str, base_dir: Path) -> str:
    spec = spec.strip()
    if not spec:
        return spec

    if spec.startswith("|") or spec.endswith("|"):
        return spec

    if ":" in spec:
        ark_path, remainder = spec.split(":", 1)
        ark_path_obj = Path(ark_path)
        if not ark_path_obj.is_absolute():
            ark_path_obj = (base_dir / ark_path_obj).resolve()
        return f"{ark_path_obj}:{remainder}"

    path_obj = Path(spec)
    if not path_obj.is_absolute():
        path_obj = (base_dir / path_obj).resolve()
    return str(path_obj)


def detect_corrupted(
    scp_path: Path, base_dir: Path, silent: bool = False
) -> Tuple[Dict[str, str], int]:
    corrupted: Dict[str, str] = {}
    total = 0

    with scp_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue

            try:
                key, spec = line.strip().split(maxsplit=1)
            except ValueError:
                corrupted[f"<malformed-{total}>"] = "Malformed line without spec."
                continue

            total += 1

            norm_spec = _resolve_spec(spec, base_dir)

            try:
                load_mat(norm_spec)
            except Exception as exc:  # noqa: BLE001 - keep raw exception text
                corrupted[key] = str(exc)
            else:
                if not silent and total % 500 == 0:
                    print(f"Checked {total} entries...", file=sys.stderr)

    return corrupted, total


def write_outputs(
    scp_path: Path,
    output_path: Path,
    corrupted: Dict[str, str],
    report_path: Path | None,
) -> None:
    keep = 0
    drop = 0

    with scp_path.open("r", encoding="utf-8") as src, output_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            key = line.strip().split(maxsplit=1)[0]
            if key in corrupted:
                drop += 1
                continue
            dst.write(line)
            keep += 1

    if report_path is not None:
        with report_path.open("w", encoding="utf-8") as rep:
            for key, err in corrupted.items():
                rep.write(f"{key}\t{err}\n")

    print(
        f"Wrote filtered wav.scp to {output_path} "
        f"(kept {keep}, removed {drop})."
    )


def main() -> None:
    args = parse_args()

    scp_path = args.scp.resolve()
    if not scp_path.exists():
        raise FileNotFoundError(f"Cannot find wav.scp at {scp_path}")

    base_dir = args.root.resolve() if args.root else scp_path.parent.resolve()
    output_path = args.output.resolve() if args.output else scp_path.with_suffix("")
    if output_path == scp_path:
        raise ValueError("Output path must differ from the input wav.scp.")

    if args.output is None:
        output_path = scp_path.parent / (scp_path.name + ".clean")

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    if args.report is not None and args.report.exists():
        raise FileExistsError(f"Report file already exists: {args.report.resolve()}")

    corrupted, total = detect_corrupted(scp_path, base_dir, silent=args.silent)

    print(f"Total entries checked: {total}")
    print(f"Corrupted entries found: {len(corrupted)}")

    write_outputs(scp_path, output_path, corrupted, args.report)


if __name__ == "__main__":
    main()









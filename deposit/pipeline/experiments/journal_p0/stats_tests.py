"""Compatibility entry point for the retired statistics script.

The former implementation treated disruptions as iid Wilcoxon observations,
reported iid intervals, and mixed R1b with R4 analyses.  Those paths are
scientifically retired.  This module can only forward an R1b input to the
scenario-clustered R5 implementation in :mod:`r5_stats`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import r5_stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e4-input", type=Path, default=None,
                        help="R1b e4_raw.csv (preferred spelling)")
    parser.add_argument("--e4-input-dir", type=Path, default=None,
                        help="legacy directory spelling; resolves to e4_raw.csv")
    parser.add_argument("--rho-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", "--out-dir", dest="output_dir",
                        type=Path, required=True)
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--snap-id", default=None)
    parser.add_argument("--holm-registration", type=Path, default=None)
    parser.add_argument("--r4-input-dir", default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--skip-e4", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-dir", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.r4_input_dir is not None or args.skip_e4 or args.input_dir is not None:
        parser.error("R4/Friedman/runtime and --skip-e4 paths are retired; run R5 from "
                     "validated R1b e4_raw.csv only")
    if (args.e4_input is None) == (args.e4_input_dir is None):
        parser.error("provide exactly one of --e4-input or --e4-input-dir")
    e4_input = (args.e4_input if args.e4_input is not None
                else args.e4_input_dir / "e4_raw.csv")
    return r5_stats.main([
        "--e4-input", str(e4_input),
        "--rho-manifest", str(args.rho_manifest),
        "--output-dir", str(args.output_dir),
        *(["--formal"] if args.formal else []),
        *(["--snap-id", str(args.snap_id)] if args.snap_id is not None else []),
        *(["--holm-registration", str(args.holm_registration)]
          if args.holm_registration is not None else []),
    ])


if __name__ == "__main__":
    raise SystemExit(main())

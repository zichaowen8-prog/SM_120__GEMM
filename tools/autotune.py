#!/usr/bin/env python3
"""Run the compiled candidate registry and rank only correctness-gated results."""
from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="32", choices=("32", "64", "128", "all"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--family", default="padding",
                        help="'padding', 'scheduling', 'warp_shapes', 'tma', an exact version, or empty for the complete registry")
    args = parser.parse_args()
    output = ROOT / "results" / f"autotune_g{args.group}.csv"
    command = [str(ROOT / "build" / "int4_gemm"), "--benchmark", "--size", "4096",
               "--group", args.group, "--warmup", str(args.warmup), "--iters", str(args.iters),
               "--csv", str(output)]
    if args.family:
        command += ["--only", args.family]
    subprocess.run(command, cwd=ROOT, check=True)
    with output.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if row["version"] not in {"quantize", "repack", "cublas_fp16", "separate"}
                and row["cache_mode"] == "steady"]
    rows.sort(key=lambda row: float(row["median_ms"]))
    print("rank,group,version,median_ms,effective_tflops,registers,smem,occupancy")
    for rank, row in enumerate(rows, 1):
        print(f"{rank},{row['group_size']},{row['version']},{row['median_ms']},{row['effective_tflops']},"
              f"{row['registers_per_thread']},{row['smem_bytes']},{row['occupancy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

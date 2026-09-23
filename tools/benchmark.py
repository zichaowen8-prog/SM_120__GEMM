#!/usr/bin/env python3
"""Reproducible launcher for the fixed RTX 5070 benchmark."""
from __future__ import annotations

import argparse
import pathlib
import subprocess


ROOT = pathlib.Path(__file__).resolve().parents[1]
BINARY = ROOT / "build" / "int4_gemm"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="all", choices=("32", "64", "128", "all"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "results" / "benchmark.csv")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    if not BINARY.exists():
        raise SystemExit(f"missing {BINARY}; run make CUDA_HOME=/usr/local/cuda first")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(BINARY), "--benchmark", "--size", "4096", "--group", args.group,
               "--warmup", str(args.warmup), "--iters", str(args.iters),
               "--csv", str(args.output)]
    if args.only:
        command += ["--only", args.only]
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())


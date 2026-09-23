#!/usr/bin/env python3
"""Extract the stable comparison metrics used by OPTIMIZATION_LOG.md from NCU reports."""
from __future__ import annotations

import argparse
import csv
import io
import pathlib
import subprocess


METRICS = {
    # Nsight Compute 2025.3 emits these raw values in ms and Kbyte,
    # respectively.  Keep the unit in the column name so downstream tables
    # cannot silently interpret 3.18 ms as 3.18 ns or 32.768 KB as bytes.
    "gpu__time_duration.sum": "duration_ms",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_active_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
    "l1tex__throughput.avg.pct_of_peak_sustained_active": "l1tex_throughput_pct",
    "lts__t_sector_hit_rate.pct": "l2_hit_pct",
    "sm__warps_active.avg.per_cycle_active": "active_warps_per_sm",
    "smsp__warps_eligible.avg.per_cycle_active": "eligible_warps_per_scheduler",
    "smsp__issue_active.avg.per_cycle_active": "issued_warps_per_scheduler",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio": "long_scoreboard",
    "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio": "short_scoreboard",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "barrier",
    "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio": "mio_throttle",
    "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio": "math_pipe",
    "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio": "not_selected",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_dynamic": "dynamic_smem_kbytes",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy_pct",
    "derived__local_spilling_requests": "local_spilling_requests",
}


def parse_report(path: pathlib.Path) -> dict[str, str]:
    result = subprocess.run(
        ["/usr/local/cuda/bin/ncu", "--import", str(path), "--page", "raw", "--csv"],
        check=True, capture_output=True, text=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if len(rows) < 3:
        raise RuntimeError(f"unexpected NCU CSV shape for {path}")
    header, _units, values = rows[:3]
    record = {"report": str(path)}
    for metric, label in METRICS.items():
        record[label] = values[header.index(metric)] if metric in header else ""
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    records = [parse_report(path) for path in args.reports]
    fields = ["report", *METRICS.values()]
    sink = io.StringIO()
    writer = csv.DictWriter(sink, fieldnames=fields)
    writer.writeheader(); writer.writerows(records)
    print(sink.getvalue(), end="")
    if args.output:
        args.output.write_text(sink.getvalue())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

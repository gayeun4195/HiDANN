#!/usr/bin/env python3
"""Summarize a HiDANN dataset run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="simplewiki")
    parser.add_argument("--memory-limit-bytes", type=int, default=860148788)
    parser.add_argument("--construction-log", type=Path, default=None)
    parser.add_argument("--search-log", type=Path, default=None)
    return parser.parse_args()


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(errors="replace")


def find_one(paths: list[Path]) -> Path | None:
    existing = [p for p in paths if p.is_file()]
    return existing[-1] if existing else None


def infer_logs(run_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    construction_log = find_one(sorted((run_dir / "full" / "construction").glob("*construction_outer.log")))
    search_log = find_one(sorted((run_dir / "full" / "search").glob("*cached_multi*.log")))
    sparse_log = find_one(sorted((run_dir / "sparse" / "construction").glob("*construction_outer.log")))
    return construction_log, search_log, sparse_log


def parse_last(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return float(matches[-1]) if matches else None


def parse_pipeline_seconds(text: str) -> float | None:
    return parse_last(r"Total Pipeline Time\s*\|\s*([0-9.]+)", text) or parse_last(
        r"Total Pipeline Time[^0-9]+([0-9.]+)", text
    )


def parse_max_rss_kb(text: str) -> int | None:
    matches = re.findall(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", text)
    return int(matches[-1]) if matches else None


def parse_elapsed(text: str) -> str:
    matches = re.findall(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*(\S+)", text)
    return matches[-1] if matches else ""


def parse_pair_selection(text: str) -> tuple[int | None, int | None]:
    matches = re.findall(r"selected_unordered_pairs=([0-9]+)\s+total_unordered_pairs=([0-9]+)", text)
    if not matches:
        return None, None
    selected, total = matches[-1]
    return int(selected), int(total)


def parse_search_rows(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 21 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "L": int(parts[0]),
                    "beam_width": int(parts[1]),
                    "qps": float(parts[2]),
                    "mean_ios": float(parts[5]),
                    "batches": float(parts[19]),
                    "recall": float(parts[20]) / 100.0,
                }
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row["L"])


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def memory_status(max_rss_kb: int | None, memory_limit_bytes: int, errors: list[str]) -> str:
    if max_rss_kb is None:
        return ""
    if max_rss_kb * 1024 > memory_limit_bytes:
        errors.append(f"construction RSS {max_rss_kb * 1024} bytes exceeds memory limit {memory_limit_bytes}")
        return "exceeded"
    return "within_limit"


def main() -> int:
    args = parse_args()
    summary_dir = args.summary_dir or (args.run_dir / "summary")
    inferred_construction, inferred_search, sparse_log = infer_logs(args.run_dir)
    construction_log = args.construction_log or inferred_construction
    search_log = args.search_log or inferred_search

    errors: list[str] = []
    warnings: list[str] = []

    construction_text = read_text(construction_log)
    search_text = read_text(search_log)
    sparse_text = read_text(sparse_log)

    construction_seconds = parse_pipeline_seconds(construction_text)
    max_rss_kb = parse_max_rss_kb(construction_text)
    if construction_seconds is None:
        errors.append("missing full construction Total Pipeline Time")
    if max_rss_kb is None:
        errors.append("missing full construction maximum RSS from /usr/bin/time")

    construction_rows: list[dict[str, object]] = []
    if construction_seconds is not None:
        construction_rows.append(
            {
                "dataset": args.dataset,
                "method": "HiDANN",
                "method_id": "HiDANN",
                "seconds": fmt(construction_seconds),
                "elapsed_time": parse_elapsed(construction_text),
                "max_rss_kb": max_rss_kb or "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": memory_status(max_rss_kb, args.memory_limit_bytes, errors),
                "include_in_primary_plot": "1",
                "source": str(construction_log or ""),
            }
        )

    sparse_seconds = parse_pipeline_seconds(sparse_text)
    selected_pairs, total_pairs = parse_pair_selection(sparse_text)
    if sparse_seconds is not None:
        construction_rows.append(
            {
                "dataset": args.dataset,
                "method": "HiDANN sparse cross-partition probing check",
                "method_id": "HiDANN_sparse_adjacent",
                "seconds": fmt(sparse_seconds),
                "elapsed_time": parse_elapsed(sparse_text),
                "max_rss_kb": parse_max_rss_kb(sparse_text) or "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": "",
                "include_in_primary_plot": "0",
                "source": str(sparse_log or ""),
            }
        )
        if selected_pairs is None or total_pairs is None:
            errors.append("sparse construction ran but pair-selection counts were not found")
        elif not (0 < selected_pairs < total_pairs):
            errors.append(f"sparse pair selection looks invalid: selected={selected_pairs}, total={total_pairs}")

    write_csv(
        summary_dir / "construction_time.csv",
        construction_rows,
        [
            "dataset",
            "method",
            "method_id",
            "seconds",
            "elapsed_time",
            "max_rss_kb",
            "memory_limit_bytes",
            "memory_status",
            "include_in_primary_plot",
            "source",
        ],
    )

    search_rows_raw = parse_search_rows(search_text)
    if not search_rows_raw:
        errors.append("missing parsed search result rows")
    search_rows = [
        {
            "dataset": args.dataset,
            "method": "HiDANN index + HiDANN search",
            "arm": "hidann_index_hidann_search",
            "L": int(row["L"]),
            "recall_at_10": fmt(row["recall"]),
            "qps": fmt(row["qps"]),
            "mean_ios": fmt(row["mean_ios"]),
            "batches": fmt(row["batches"]),
            "beam_width": int(row["beam_width"]),
            "source": str(search_log or ""),
        }
        for row in search_rows_raw
    ]
    write_csv(
        summary_dir / "search_full_system.csv",
        search_rows,
        [
            "dataset",
            "method",
            "arm",
            "L",
            "recall_at_10",
            "qps",
            "mean_ios",
            "batches",
            "beam_width",
            "source",
        ],
    )

    status = "error" if errors else ("warning" if warnings else "ok")
    lines = [
        f"status={status}",
        f"run_dir={args.run_dir}",
        "",
        "construction:",
        f"  seconds={fmt(construction_seconds)}",
        f"  max_rss_kb={max_rss_kb or ''}",
        "",
        "search rows:",
    ]
    for row in search_rows:
        if int(row["L"]) in {50, 100, 500}:
            lines.append(f"  L={row['L']}: recall={row['recall_at_10']} qps={row['qps']}")
    if warnings:
        lines += ["", "warnings:"] + [f"  - {item}" for item in warnings]
    if errors:
        lines += ["", "errors:"] + [f"  - {item}" for item in errors]

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "run_report.txt").write_text("\n".join(lines) + "\n")
    (summary_dir / "pipeline_status.json").write_text(
        json.dumps({"status": status, "errors": errors, "warnings": warnings}, indent=2) + "\n"
    )
    print("\n".join(lines))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

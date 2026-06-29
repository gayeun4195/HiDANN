#!/usr/bin/env python3
"""Collect B=1 construction-quality measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "dataset",
    "method",
    "method_id",
    "L",
    "recall_at_10",
    "one_minus_recall_at_10",
    "avg_hop",
    "avg_cmps",
    "qps",
    "threads",
    "entry_protocol",
    "beam_width",
    "start_node",
    "metric_source",
    "index_path",
    "source",
]

METHOD_ORDER = {
    "Vamana_alpha1.2": 0,
    "DiskANN": 1,
    "SOGAIC": 2,
    "HiDANN": 3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path)
    parser.add_argument("--out-csv", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float | str | None, digits: int = 6) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def normalize_row(row: dict[str, str], source: Path) -> dict[str, str]:
    out = {field: row.get(field, "") for field in FIELDS}
    out["dataset"] = out["dataset"] or "simplewiki"
    out["recall_at_10"] = fmt(float(out["recall_at_10"]))
    out["one_minus_recall_at_10"] = fmt(max(float(out["one_minus_recall_at_10"]), 1e-9))
    out["avg_hop"] = fmt(float(out["avg_hop"]))
    if out["avg_cmps"]:
        out["avg_cmps"] = fmt(float(out["avg_cmps"]))
    if out["qps"]:
        out["qps"] = fmt(float(out["qps"]))
    out["source"] = out["source"] or str(source)
    return out


def row_key(row: dict[str, str]) -> tuple[str, str, int]:
    return (row["dataset"], row["method_id"], int(float(row["L"])))


def sort_key(row: dict[str, str]) -> tuple[int, str, int]:
    return (METHOD_ORDER.get(row["method_id"], 99), row["method_id"], int(float(row["L"])))


def main() -> int:
    args = parse_args()
    quality_dir = args.quality_dir or (args.run_dir / "quality")
    out_csv = args.out_csv or (args.run_dir / "summary" / "index_quality.csv")

    rows_by_key: dict[tuple[str, str, int], dict[str, str]] = {}
    for path in sorted(quality_dir.glob("*_memory_quality.csv")):
        for row in read_csv(path):
            normalized = normalize_row(row, path)
            rows_by_key[row_key(normalized)] = normalized

    rows = sorted(rows_by_key.values(), key=sort_key)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    report_lines = [
        f"quality_rows={len(rows)}",
        f"quality_csv={out_csv}",
        f"quality_dir={quality_dir}",
        "",
        "checkpoints:",
    ]
    by_method_l = {(row["method_id"], int(float(row["L"]))): row for row in rows}
    for l_value in [50, 100, 500]:
        report_lines.append(f"  L={l_value}:")
        for method_id in ["DiskANN", "SOGAIC", "HiDANN"]:
            row = by_method_l.get((method_id, l_value))
            if row is None:
                report_lines.append(f"    {method_id}: missing")
            else:
                report_lines.append(
                    f"    {method_id}: recall={float(row['recall_at_10']):.4f} "
                    f"error={float(row['one_minus_recall_at_10']):.4f} "
                    f"avg_hop={float(row['avg_hop']):.2f} start_node={row['start_node']} "
                    f"source={row['metric_source']}"
                )

    missing = [method_id for method_id in ["DiskANN", "SOGAIC", "HiDANN"] if not any(row["method_id"] == method_id for row in rows)]
    if missing:
        report_lines += ["", "warnings:"]
        report_lines.extend(f"  - missing quality rows for {method_id}" for method_id in missing)

    report_path = out_csv.parent / "quality_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n")
    print("\n".join(report_lines))
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

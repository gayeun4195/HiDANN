#!/usr/bin/env python3
"""Summarize a HiDANN/baseline comparison run."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SEARCH_METHODS = {
    "diskann_index_diskann_search": "DiskANN index + DiskANN search",
    "diskann_index_pipeann_search": "DiskANN index + PipeANN search",
    "hidann_index_hidann_search": "HiDANN index + HiDANN search",
    "hidann_index_diskann_search": "HiDANN index + DiskANN search",
    "hidann_index_pipeann_search": "HiDANN index + PipeANN search",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--summary-dir", type=Path, default=None)
    parser.add_argument("--dataset", default="simplewiki")
    parser.add_argument("--dataset-prefix", default=None)
    parser.add_argument("--memory-limit-bytes", type=int, default=860148788)
    return parser.parse_args()


def read_text(path: Path | None) -> str:
    if path is None or not path.is_file():
        return ""
    return path.read_text(errors="replace")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


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


def parse_last(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        return None
    value = matches[-1]
    if isinstance(value, tuple):
        value = value[-1]
    return float(value)


def parse_max_rss_kb(text: str) -> int | None:
    matches = re.findall(r"Maximum resident set size \(kbytes\):\s*([0-9]+)", text)
    return max(int(match) for match in matches) if matches else None


def format_memory_status(max_rss_kb: float | int | None, memory_limit_bytes: int) -> str:
    if max_rss_kb is None:
        return ""
    return "within_limit" if int(max_rss_kb) * 1024 <= memory_limit_bytes else "exceeded"


def parse_hidann_pq_seconds(hidann_dir: Path) -> float | None:
    for path in sorted((hidann_dir / "full" / "construction").glob("*_pipeline.log")):
        text = read_text(path)
        for pattern in [
            r"PQ Generation=([0-9.]+)s",
            r"PQ Generation\s*\|\s*([0-9.]+)",
            r"pq_gen=([0-9.]+)\s*s",
            r"Pruning 1st.*?PQ.*?([0-9.]+)",
        ]:
            values = [float(match) for match in re.findall(pattern, text, flags=re.MULTILINE)]
            positives = [value for value in values if value > 0]
            if positives:
                return max(positives)
    return None


def load_hidann_construction(hidann_dir: Path) -> tuple[float | None, float | None, int | None]:
    rows = read_csv(hidann_dir / "summary" / "construction_time.csv")
    total = None
    sparse = None
    max_rss = None
    for row in rows:
        if row.get("method_id") == "HiDANN":
            total = float(row["seconds"])
            if row.get("max_rss_kb"):
                max_rss = int(row["max_rss_kb"])
        elif row.get("method_id") == "HiDANN_sparse_adjacent":
            sparse = float(row["seconds"])
    if total is None:
        logs = sorted((hidann_dir / "full" / "construction").glob("*construction_outer.log"))
        text = read_text(logs[-1] if logs else None)
        total = parse_last(r"Total Pipeline Time\s*\|\s*([0-9.]+)", text)
        max_rss = parse_max_rss_kb(text)
    return total, sparse, max_rss


def parse_diskann_construction(log_path: Path) -> dict[str, float | int | None]:
    text = read_text(log_path)
    return {
        "pq": parse_last(r"Time for generating quantized data:\s*([0-9.]+)", text),
        "merged_vamana": parse_last(r"Time for building merged vamana index:\s*([0-9.]+)", text),
        "max_rss_kb": parse_max_rss_kb(text),
    }


def parse_sogaic_construction(log_path: Path) -> dict[str, float | int | None]:
    text = read_text(log_path)
    return {
        "total": parse_last(r"TOTAL PIPELINE TIME:\s*([0-9.]+)\s*s", text),
        "max_rss_kb": parse_max_rss_kb(text),
    }


def parse_diskann_search_rows(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 9 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "L": int(parts[0]),
                    "beam_width": int(parts[1]),
                    "qps": float(parts[2]),
                    "mean_ios": float(parts[5]),
                    "recall": float(parts[8]) / 100.0,
                }
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row["L"])


def parse_pipeann_search_rows(text: str) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 8 or not parts[0].isdigit():
            continue
        try:
            rows.append(
                {
                    "L": int(parts[0]),
                    "beam_width": int(parts[1]),
                    "qps": float(parts[2]),
                    "mean_ios": float(parts[6]),
                    "recall": float(parts[7]) / 100.0,
                }
            )
        except ValueError:
            continue
    return sorted(rows, key=lambda row: row["L"])


def load_hidann_search_rows(hidann_dir: Path, dataset: str) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in read_csv(hidann_dir / "summary" / "search_full_system.csv"):
        if row.get("arm") != "hidann_index_hidann_search":
            continue
        out.append(
            {
                "dataset": dataset,
                "method": SEARCH_METHODS["hidann_index_hidann_search"],
                "arm": "hidann_index_hidann_search",
                "L": int(row["L"]),
                "recall_at_10": float(row["recall_at_10"]),
                "qps": float(row["qps"]),
                "mean_ios": float(row.get("mean_ios") or 0),
                "beam_width": int(float(row.get("beam_width") or 8)),
                "source": row.get("source", ""),
            }
        )
    return sorted(out, key=lambda row: int(row["L"]))


def write_search_rows(path: Path, rows: list[dict[str, object]]) -> None:
    write_csv(
        path,
        sorted(rows, key=lambda row: (str(row["arm"]), int(row["L"]))),
        [
            "dataset",
            "method",
            "arm",
            "L",
            "recall_at_10",
            "qps",
            "mean_ios",
            "beam_width",
            "source",
        ],
    )


def add_search_rows(
    rows: list[dict[str, object]],
    dataset: str,
    arm: str,
    log_path: Path,
    parser,
) -> None:
    for row in parser(read_text(log_path)):
        rows.append(
            {
                "dataset": dataset,
                "method": SEARCH_METHODS[arm],
                "arm": arm,
                "L": int(row["L"]),
                "recall_at_10": fmt(float(row["recall"])),
                "qps": fmt(float(row["qps"])),
                "mean_ios": fmt(float(row["mean_ios"])),
                "beam_width": int(row["beam_width"]),
                "source": str(log_path),
            }
        )


def checkpoint_lines(rows: list[dict[str, object]], arms: list[str], title: str) -> list[str]:
    by_arm_l = {(str(row["arm"]), int(row["L"])): row for row in rows}
    lines = ["", title]
    for l_value in [50, 100, 500]:
        lines.append(f"  L={l_value}:")
        for arm in arms:
            row = by_arm_l.get((arm, l_value))
            if row is None:
                lines.append(f"    {arm}: missing")
            else:
                lines.append(
                    f"    {arm}: recall={float(row['recall_at_10']):.4f} qps={float(row['qps']):.2f}"
                )
    return lines


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    dataset = args.dataset
    dataset_prefix = args.dataset_prefix or dataset
    summary_dir = args.summary_dir or (run_dir / "summary")
    hidann_dir = run_dir / "hidann"
    if not hidann_dir.is_dir() and (run_dir / "full" / "construction").is_dir():
        hidann_dir = run_dir

    warnings: list[str] = []
    errors: list[str] = []

    hidann_total, hidann_sparse, hidann_rss_kb = load_hidann_construction(hidann_dir)
    hidann_pq = parse_hidann_pq_seconds(hidann_dir)
    if hidann_total is None:
        errors.append("missing HiDANN construction result")
    if hidann_pq is None:
        warnings.append("missing HiDANN PQ generation time; common-PQ construction rows use available native timings")

    diskann_log = run_dir / "baselines" / "diskann" / "construction" / f"{dataset_prefix}_diskann_common_pq.log"
    sogaic_outer_log = run_dir / "baselines" / "sogaic" / "construction" / f"{dataset_prefix}_sogaic_outer.log"
    sogaic_inner_log = run_dir / "baselines" / "sogaic" / "construction" / f"{dataset_prefix}_sogaic.log"
    sogaic_source_log = sogaic_outer_log if sogaic_outer_log.is_file() else sogaic_inner_log
    diskann = parse_diskann_construction(diskann_log)
    sogaic = parse_sogaic_construction(sogaic_source_log)

    construction_rows: list[dict[str, object]] = []
    if sogaic["total"] is not None:
        rss_kb = sogaic["max_rss_kb"]
        construction_rows.append(
            {
                "dataset": dataset,
                "method": "SOGAIC common-PQ",
                "method_id": "SOGAIC_common_pq",
                "seconds": fmt(float(sogaic["total"]) + (hidann_pq or 0.0)),
                "elapsed_time": "",
                "max_rss_kb": rss_kb or "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": format_memory_status(rss_kb, args.memory_limit_bytes),
                "include_in_primary_plot": "1",
                "source": str(sogaic_source_log),
            }
        )
    else:
        warnings.append("missing SOGAIC construction result")

    if diskann["merged_vamana"] is not None:
        rss_kb = diskann["max_rss_kb"]
        common_pq_seconds = hidann_pq if hidann_pq is not None else float(diskann["pq"] or 0.0)
        construction_rows.append(
            {
                "dataset": dataset,
                "method": "DiskANN common-PQ",
                "method_id": "DiskANN_common_pq",
                "seconds": fmt(float(diskann["merged_vamana"]) + common_pq_seconds),
                "elapsed_time": "",
                "max_rss_kb": rss_kb or "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": format_memory_status(rss_kb, args.memory_limit_bytes),
                "include_in_primary_plot": "1",
                "source": str(diskann_log),
            }
        )
    else:
        warnings.append("missing DiskANN construction result")

    if hidann_total is not None:
        memory_status = format_memory_status(hidann_rss_kb, args.memory_limit_bytes)
        if memory_status == "exceeded":
            errors.append("HiDANN construction RSS exceeds configured memory limit")
        construction_rows.append(
            {
                "dataset": dataset,
                "method": "HiDANN",
                "method_id": "HiDANN",
                "seconds": fmt(hidann_total),
                "elapsed_time": "",
                "max_rss_kb": hidann_rss_kb or "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": memory_status,
                "include_in_primary_plot": "1",
                "source": str(hidann_dir / "summary" / "construction_time.csv"),
            }
        )

    if hidann_sparse is not None:
        construction_rows.append(
            {
                "dataset": dataset,
                "method": "HiDANN sparse cross-partition probing",
                "method_id": "HiDANN_sparse_adjacent",
                "seconds": fmt(hidann_sparse),
                "elapsed_time": "",
                "max_rss_kb": "",
                "memory_limit_bytes": args.memory_limit_bytes,
                "memory_status": "",
                "include_in_primary_plot": "0",
                "source": str(hidann_dir / "summary" / "construction_time.csv"),
            }
        )

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

    search_rows: list[dict[str, object]] = []
    search_rows.extend(load_hidann_search_rows(hidann_dir, dataset))
    diskann_search_log = run_dir / "baselines" / "diskann" / "search" / f"{dataset_prefix}_diskann_search.log"
    pipeann_search_log = run_dir / "baselines" / "pipeann" / "search" / f"{dataset_prefix}_pipeann_search.log"
    add_search_rows(search_rows, dataset, "diskann_index_diskann_search", diskann_search_log, parse_diskann_search_rows)
    add_search_rows(search_rows, dataset, "diskann_index_pipeann_search", pipeann_search_log, parse_pipeann_search_rows)

    if not any(row["arm"] == "hidann_index_hidann_search" for row in search_rows):
        errors.append("missing HiDANN search rows")
    if not any(row["arm"] == "diskann_index_diskann_search" for row in search_rows):
        warnings.append("missing DiskANN search rows")
    if not any(row["arm"] == "diskann_index_pipeann_search" for row in search_rows):
        warnings.append("missing PipeANN search rows")
    write_search_rows(summary_dir / "search_full_system.csv", search_rows)

    same_index_rows: list[dict[str, object]] = []
    same_index_rows.extend(load_hidann_search_rows(hidann_dir, dataset))
    same_diskann_log = run_dir / "same_hidann_index" / "diskann_search" / f"{dataset_prefix}_hidann_index_diskann_search.log"
    same_pipeann_log = run_dir / "same_hidann_index" / "pipeann_search" / f"{dataset_prefix}_hidann_index_pipeann_search.log"
    add_search_rows(
        same_index_rows,
        dataset,
        "hidann_index_diskann_search",
        same_diskann_log,
        parse_diskann_search_rows,
    )
    add_search_rows(
        same_index_rows,
        dataset,
        "hidann_index_pipeann_search",
        same_pipeann_log,
        parse_pipeann_search_rows,
    )
    if same_index_rows:
        write_search_rows(summary_dir / "search_same_hidann_index.csv", same_index_rows)

    status = "error" if errors else ("warning" if warnings else "ok")
    lines = [
        f"status={status}",
        f"run_dir={run_dir}",
        f"dataset={dataset}",
        "",
        "construction primary rows:",
    ]
    for row in construction_rows:
        if row["include_in_primary_plot"] == "1":
            lines.append(
                f"  {row['method_id']}: seconds={row['seconds']} max_rss_kb={row['max_rss_kb']} "
                f"memory_status={row['memory_status']}"
            )
    if hidann_sparse is not None:
        lines.append(f"  HiDANN_sparse_adjacent: seconds={hidann_sparse:.6f}")

    lines.extend(
        checkpoint_lines(
            search_rows,
            [
                "diskann_index_diskann_search",
                "diskann_index_pipeann_search",
                "hidann_index_hidann_search",
            ],
            "full-system search checkpoints:",
        )
    )
    if same_index_rows:
        lines.extend(
            checkpoint_lines(
                same_index_rows,
                [
                    "hidann_index_diskann_search",
                    "hidann_index_pipeann_search",
                    "hidann_index_hidann_search",
                ],
                "same-HiDANN-index search checkpoints:",
            )
        )
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

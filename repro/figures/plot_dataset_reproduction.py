#!/usr/bin/env python3
"""Draw single-panel HiDANN reproduction figures as SVG."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent
DEFAULT_PLOTS = ROOT / "plots"

TICK_FONT = 17.0
AXIS_FONT = 19.0
PANEL_FONT = 23.0
LEGEND_FONT = 23.0
PANEL_STROKE = 2.0
SERIES_STROKE = 2.15
GRID_MAJOR_STROKE = 0.65
GRID_MINOR_STROKE = 0.45
POINT_MARKER_SIZE = 4.0
LEGEND_MARKER_SIZE = 4.5
QPS_SCALE = 10000.0

QUALITY_STYLE = {
    "Vamana_alpha1.2": {"label": "Vamana", "color": "#FFA500", "marker": "right"},
    "DiskANN": {"label": "DiskANN", "color": "#CC0000", "marker": "square"},
    "SOGAIC": {"label": "SOGAIC-style", "color": "#556B2F", "marker": "star"},
    "HiDANN": {"label": "HiDANN", "color": "#001F3F", "marker": "circle"},
}

SEARCH_STYLE = {
    "diskann_index_diskann_search": {"label": "DiskANN", "color": "#CC0000", "marker": "square"},
    "diskann_index_pipeann_search": {"label": "PipeANN", "color": "#A9A9A9", "marker": "square"},
    "hidann_index_hidann_search": {"label": "HiDANN", "color": "#001F3F", "marker": "circle"},
    "hidann_index_diskann_search": {"label": "DiskANN", "color": "#CC0000", "marker": "square"},
    "hidann_index_pipeann_search": {"label": "PipeANN", "color": "#A9A9A9", "marker": "square"},
}

CONSTRUCTION_EFFECT_STYLE = {
    "diskann_index_pipeann_search": {"label": "PipeANN", "color": "#A9A9A9", "marker": "square"},
    "diskann_index_diskann_search": {"label": "DiskANN", "color": "#CC0000", "marker": "square"},
    "hidann_index_diskann_search": {"label": "HiDANN (C)", "color": "#001F3F", "marker": "circle"},
}

BAR_STYLE = {
    "SOGAIC_common_pq": "#556B2F",
    "DiskANN_common_pq": "#CC0000",
    "HiDANN": "#001F3F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOTS)
    parser.add_argument("--width", type=float, default=620.0)
    parser.add_argument("--height", type=float, default=390.0)
    parser.add_argument("--dataset-label", default="SimpleWiki")
    parser.add_argument("--dataset-slug", default=None)
    parser.add_argument(
        "--search-min-l",
        type=float,
        default=30.0,
        help="Minimum search L shown in Recall@10/QPS plots; matches the paper search-figure crop by default.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_svg(path: Path, body: str, width: float, height: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
                '<rect width="100%" height="100%" fill="white"/>',
                '<style>text{font-family:Calibri,Arial,Helvetica,sans-serif;fill:#111}</style>',
                '<g fill="#111">',
                body,
                "</g>",
                "</svg>",
            ]
        )
        + "\n"
    )
    print(f"wrote {path}")


def dataset_slug(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or "dataset"


def plot_path(out_dir: Path, dataset_key: str, paper_id: str, stem: str) -> Path:
    return out_dir / f"{paper_id}_{stem}_{dataset_key}.svg"


def font(value: float) -> str:
    return f"{value:.1f}"


def marker(kind: str, x: float, y: float, color: str, size: float = POINT_MARKER_SIZE) -> str:
    if kind == "circle":
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}"/>'
    if kind == "square":
        return f'<rect x="{x-size:.2f}" y="{y-size:.2f}" width="{2*size:.2f}" height="{2*size:.2f}" fill="{color}"/>'
    if kind == "diamond":
        s = size * 1.15
        pts = [(x, y - s), (x + s, y), (x, y + s), (x - s, y)]
        return f'<polygon points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in pts)}" fill="{color}"/>'
    if kind == "right":
        s = size * 1.25
        pts = [(x + s, y), (x - s * 0.8, y - s), (x - s * 0.8, y + s)]
        return f'<polygon points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in pts)}" fill="{color}"/>'
    if kind == "star":
        pts = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            radius = size * (1.25 if i % 2 == 0 else 0.55)
            pts.append((x + radius * math.cos(angle), y + radius * math.sin(angle)))
        return f'<polygon points="{" ".join(f"{px:.2f},{py:.2f}" for px, py in pts)}" fill="{color}"/>'
    return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{size:.2f}" fill="{color}"/>'


def log_ticks(vmin: float, vmax: float) -> list[float]:
    lo = math.floor(math.log10(vmin))
    hi = math.ceil(math.log10(vmax))
    return [10.0**p for p in range(lo, hi + 1) if vmin <= 10.0**p <= vmax]


def log_minor_ticks(vmin: float, vmax: float) -> list[float]:
    lo = math.floor(math.log10(vmin)) - 1
    hi = math.ceil(math.log10(vmax)) + 1
    out: list[float] = []
    for power in range(lo, hi + 1):
        base = 10.0**power
        for mult in range(2, 10):
            tick = mult * base
            if vmin <= tick <= vmax:
                out.append(tick)
    return out


def log_tick_text(x: float, y: float, value: float, anchor: str = "middle") -> str:
    power = round(math.log10(value))
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" font-size="{font(TICK_FONT)}">10<tspan dy="-7" font-size="{font(TICK_FONT * 0.72)}">{power}</tspan></text>'


def nice_step(raw: float) -> float:
    if raw <= 0:
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for mult in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        step = mult * power
        if step >= raw:
            return step
    return 10.0 * power


def linear_ticks(vmin: float, vmax: float, target: int = 4) -> list[float]:
    if vmax <= vmin:
        return [vmin]
    step = nice_step((vmax - vmin) / max(target - 1, 1))
    start = math.ceil(vmin / step) * step
    ticks: list[float] = []
    value = start
    while value <= vmax + step * 0.25:
        if vmin - step * 0.1 <= value <= vmax + step * 0.1:
            ticks.append(round(value, 10))
        value += step
    return ticks or [vmin, vmax]


def qps_ticks(max_qps: float) -> tuple[list[float], float]:
    ymax = max_qps / QPS_SCALE * 1.10
    step = nice_step(ymax / 3.0)
    n = max(3, math.ceil(ymax / step))
    ticks = [i * step for i in range(n + 1)]
    return ticks, max(ticks)


def fmt_recall(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def fmt_axis(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def line_path(points: list[tuple[float, float]]) -> str:
    return " ".join(("M" if i == 0 else "L") + f" {x:.2f} {y:.2f}" for i, (x, y) in enumerate(points))


def legend(items: list[tuple[str, str, str]], x: float, y: float) -> str:
    out: list[str] = []
    cur_x = x
    for label, color, shape in items:
        out.append(f'<line x1="{cur_x:.2f}" y1="{y - 4:.2f}" x2="{cur_x + 24:.2f}" y2="{y - 4:.2f}" stroke="{color}" stroke-width="2.2"/>')
        out.append(marker(shape, cur_x + 12, y - 4, color, LEGEND_MARKER_SIZE))
        out.append(f'<text x="{cur_x + 34:.2f}" y="{y + 2:.2f}" font-size="{font(LEGEND_FONT)}" font-weight="700">{escape(label)}</text>')
        cur_x += 92 + len(label) * 10.5
    return "\n".join(out)


def draw_quality(results_dir: Path, out_dir: Path, width: float, height: float, dataset_label: str, dataset_key: str) -> None:
    csv_path = results_dir / "index_quality.csv"
    if not csv_path.is_file():
        print(f"skipping index-quality plot; missing {csv_path}")
        return
    rows = read_csv(csv_path)
    data: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        if row.get("entry_protocol") and row.get("entry_protocol") != "single_common_entry":
            continue
        data[row["method_id"]].append(
            {
                "error": max(float(row["one_minus_recall_at_10"]), 1e-9),
                "hop": max(float(row["avg_hop"]), 1e-9),
            }
        )
    for points in data.values():
        points.sort(key=lambda p: p["error"], reverse=True)

    if not any(data.values()):
        print(f"skipping index-quality plot; no single_common_entry rows in {csv_path}")
        return

    ml, mr, mt, mb = 76.0, 22.0, 52.0, 86.0
    x0, y0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb
    all_errors = [p["error"] for points in data.values() for p in points]
    lx_min, lx_max = math.log10(min(all_errors)) - 0.08, math.log10(max(all_errors)) + 0.08
    ly_min, ly_max = math.log10(10.0), math.log10(1000.0)

    def sx(value: float) -> float:
        return x0 + (lx_max - math.log10(value)) / (lx_max - lx_min) * pw

    def sy(value: float) -> float:
        return y0 + ph - (math.log10(value) - ly_min) / (ly_max - ly_min) * ph

    out = [f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="white" stroke="#111" stroke-width="{font(PANEL_STROKE)}"/>']
    for tick in log_minor_ticks(10.0**lx_min, 10.0**lx_max):
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#e7e7e7" stroke-width="{font(GRID_MINOR_STROKE)}" opacity="0.62"/>')
    for tick in log_minor_ticks(10.0**ly_min, 10.0**ly_max):
        out.append(f'<line x1="{x0:.2f}" y1="{sy(tick):.2f}" x2="{x0+pw:.2f}" y2="{sy(tick):.2f}" stroke="#e7e7e7" stroke-width="{font(GRID_MINOR_STROKE)}" opacity="0.62"/>')
    for tick in log_ticks(10.0**lx_min, 10.0**lx_max):
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#d8d8d8" stroke-width="{font(GRID_MAJOR_STROKE)}" opacity="0.82"/>')
        out.append(log_tick_text(sx(tick), y0 + ph + 24, tick))
    for tick in log_ticks(10.0**ly_min, 10.0**ly_max):
        out.append(f'<line x1="{x0:.2f}" y1="{sy(tick):.2f}" x2="{x0+pw:.2f}" y2="{sy(tick):.2f}" stroke="#d8d8d8" stroke-width="{font(GRID_MAJOR_STROKE)}" opacity="0.82"/>')
        out.append(log_tick_text(x0 - 8, sy(tick) + 5, tick, "end"))

    for method_id, style in QUALITY_STYLE.items():
        xy = [(sx(p["error"]), sy(p["hop"])) for p in data.get(method_id, []) if 10.0 <= p["hop"] <= 1000.0]
        if len(xy) >= 2:
            out.append(f'<path d="{line_path(xy)}" fill="none" stroke="{style["color"]}" stroke-width="{font(SERIES_STROKE)}"/>')
        for x, y in xy:
            out.append(marker(style["marker"], x, y, style["color"], POINT_MARKER_SIZE))

    present = [QUALITY_STYLE[m] for m in QUALITY_STYLE if data.get(m)]
    out.append(legend([(s["label"], s["color"], s["marker"]) for s in present], x0 + 2, 25))
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 38:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700">1 - Recall@10</text>')
    out.append(f'<text x="24" y="{y0 + ph / 2:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700" transform="rotate(-90 24 {y0 + ph / 2:.2f})">Average hop #</text>')
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 9:.2f}" text-anchor="middle" font-size="{font(PANEL_FONT)}" font-weight="700">{escape(dataset_label)}</text>')
    write_svg(
        plot_path(out_dir, dataset_key, "fig9", "main_index_quality"),
        "\n".join(out),
        width,
        height,
    )


def draw_construction(results_dir: Path, out_dir: Path, width: float, height: float, dataset_label: str, dataset_key: str) -> None:
    csv_path = results_dir / "construction_time.csv"
    if not csv_path.is_file():
        print(f"skipping construction plot; missing {csv_path}")
        return
    rows = [r for r in read_csv(csv_path) if r.get("include_in_primary_plot", "1") == "1"]
    if not rows:
        print(f"skipping construction plot; no plottable rows in {csv_path}")
        return
    ml, mr, mt, mb = 72.0, 24.0, 42.0, 86.0
    x0, y0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb
    values = [float(r["seconds"]) for r in rows]
    ly_min, ly_max = math.log10(100.0), math.log10(max(values) * 1.45)

    def sy(value: float) -> float:
        return y0 + ph - (math.log10(value) - ly_min) / (ly_max - ly_min) * ph

    out = [f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="white" stroke="#111" stroke-width="1.6"/>']
    for tick in log_minor_ticks(100.0, max(values) * 1.45):
        out.append(f'<line x1="{x0:.2f}" y1="{sy(tick):.2f}" x2="{x0+pw:.2f}" y2="{sy(tick):.2f}" stroke="#e8e8e8" stroke-width="0.5"/>')
    for tick in log_ticks(100.0, max(values) * 1.45):
        out.append(f'<line x1="{x0:.2f}" y1="{sy(tick):.2f}" x2="{x0+pw:.2f}" y2="{sy(tick):.2f}" stroke="#d0d0d0" stroke-width="0.7"/>')
        out.append(log_tick_text(x0 - 8, sy(tick) + 4, tick, "end"))

    gap = 36.0
    bar_w = (pw - gap * (len(rows) + 1)) / len(rows)
    fallback_colors = ["#00897B", "#CC0000", "#001F3F", "#7B3F00", "#6A1B9A"]
    for i, row in enumerate(rows):
        value = float(row["seconds"])
        x = x0 + gap + i * (bar_w + gap)
        y = sy(value)
        color = BAR_STYLE.get(row["method_id"], fallback_colors[i % len(fallback_colors)])
        out.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{y0+ph-y:.2f}" fill="{color}"/>')
        out.append(f'<text x="{x + bar_w / 2:.2f}" y="{y - 7:.2f}" text-anchor="middle" font-size="13" font-weight="700">{value:.1f}s</text>')
        label = escape(row["method"].replace(" common-PQ", ""))
        out.append(f'<text x="{x + bar_w / 2:.2f}" y="{y0 + ph + 23:.2f}" text-anchor="middle" font-size="13" font-weight="700">{label}</text>')

    out.append(f'<text x="24" y="{y0 + ph / 2:.2f}" text-anchor="middle" font-size="15" font-weight="700" transform="rotate(-90 24 {y0 + ph / 2:.2f})">Construction time (s, log)</text>')
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 12:.2f}" text-anchor="middle" font-size="14" font-weight="700">{escape(dataset_label)} Construction Comparison</text>')
    write_svg(
        plot_path(out_dir, dataset_key, "table3", "construction_time"),
        "\n".join(out),
        width,
        height,
    )


def draw_construction_effect(results_dir: Path, out_dir: Path, width: float, height: float, search_min_l: float, dataset_label: str, dataset_key: str) -> None:
    full_csv = results_dir / "search_full_system.csv"
    same_csv = results_dir / "search_same_hidann_index.csv"
    if not full_csv.is_file() or not same_csv.is_file():
        print(f"skipping construction-effect plot; missing {full_csv} or {same_csv}")
        return

    wanted = {
        "diskann_index_pipeann_search": full_csv,
        "diskann_index_diskann_search": full_csv,
        "hidann_index_diskann_search": same_csv,
    }
    data: dict[str, list[dict[str, float]]] = defaultdict(list)
    for csv_path in sorted(set(wanted.values())):
        for row in read_csv(csv_path):
            arm = row["arm"]
            if wanted.get(arm) != csv_path:
                continue
            l_value = float(row["L"])
            if l_value < search_min_l:
                continue
            data[arm].append({"L": l_value, "qps": float(row["qps"]), "recall": float(row["recall_at_10"])})
    for points in data.values():
        points.sort(key=lambda p: p["recall"])

    if not any(data.values()):
        print(f"skipping construction-effect plot; no rows with L >= {search_min_l:g}")
        return

    ml, mr, mt, mb = 76.0, 22.0, 52.0, 86.0
    x0, y0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb
    all_qps = [p["qps"] for points in data.values() for p in points]
    all_recalls = [p["recall"] for points in data.values() for p in points]
    rx_min = max(0.0, min(all_recalls) - max((max(all_recalls) - min(all_recalls)) * 0.08, 0.006))
    rx_max = min(1.0, max(all_recalls) + max((max(all_recalls) - min(all_recalls)) * 0.03, 0.003))
    if rx_max - rx_min < 0.03:
        rx_min = max(0.0, rx_max - 0.03)
    ytick_values, ytop = qps_ticks(max(all_qps))

    def sx(value: float) -> float:
        return x0 + (value - rx_min) / (rx_max - rx_min) * pw

    def sy(value: float) -> float:
        return y0 + ph - (value / QPS_SCALE) / ytop * ph

    out = [f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="white" stroke="#111" stroke-width="{font(PANEL_STROKE)}"/>']
    for tick in linear_ticks(rx_min, rx_max):
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#d4d4d4" stroke-width="0.6" opacity="0.70"/>')
        out.append(f'<text x="{sx(tick):.2f}" y="{y0 + ph + 24:.2f}" text-anchor="middle" font-size="{font(TICK_FONT)}">{escape(fmt_recall(tick))}</text>')
    for tick in ytick_values:
        out.append(f'<line x1="{x0:.2f}" y1="{y0 + ph - tick / ytop * ph:.2f}" x2="{x0+pw:.2f}" y2="{y0 + ph - tick / ytop * ph:.2f}" stroke="#d4d4d4" stroke-width="0.6" opacity="0.70"/>')
        out.append(f'<text x="{x0 - 6:.2f}" y="{y0 + ph - tick / ytop * ph + 5:.2f}" text-anchor="end" font-size="{font(TICK_FONT)}">{escape(fmt_axis(tick))}</text>')

    arm_order = ["diskann_index_pipeann_search", "diskann_index_diskann_search", "hidann_index_diskann_search"]
    for arm in arm_order:
        style = CONSTRUCTION_EFFECT_STYLE[arm]
        xy = [(sx(p["recall"]), sy(p["qps"])) for p in data.get(arm, [])]
        if len(xy) >= 2:
            out.append(f'<path d="{line_path(xy)}" fill="none" stroke="{style["color"]}" stroke-width="{font(SERIES_STROKE)}"/>')
        for x, y in xy:
            out.append(marker(style["marker"], x, y, style["color"], POINT_MARKER_SIZE))

    present = [arm for arm in arm_order if data.get(arm)]
    out.append(legend([(CONSTRUCTION_EFFECT_STYLE[a]["label"], CONSTRUCTION_EFFECT_STYLE[a]["color"], CONSTRUCTION_EFFECT_STYLE[a]["marker"]) for a in present], x0 + 22, 25))
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 38:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700">Recall@10</text>')
    out.append(f'<text x="24" y="{y0 + ph / 2:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700" transform="rotate(-90 24 {y0 + ph / 2:.2f})">QPS (&#215;10^4)</text>')
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 9:.2f}" text-anchor="middle" font-size="{font(PANEL_FONT)}" font-weight="700">{escape(dataset_label)}</text>')
    write_svg(
        plot_path(out_dir, dataset_key, "fig7", "construction_effect_qps"),
        "\n".join(out),
        width,
        height,
    )


def draw_same_index_search(results_dir: Path, out_dir: Path, width: float, height: float, search_min_l: float, dataset_label: str, dataset_key: str) -> None:
    csv_path = results_dir / "search_same_hidann_index.csv"
    if not csv_path.is_file():
        print(f"skipping same-index search plot; missing {csv_path}")
        return
    rows = read_csv(csv_path)
    data: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        l_value = float(row["L"])
        if l_value < search_min_l:
            continue
        recall = float(row["recall_at_10"])
        data[row["arm"]].append(
            {
                "L": l_value,
                "qps": float(row["qps"]),
                "recall": recall,
                "error": max(1.0 - recall, 1e-9),
            }
        )
    for points in data.values():
        points.sort(key=lambda p: p["error"], reverse=True)

    if not any(data.values()):
        print(f"skipping same-index search plot; no rows with L >= {search_min_l:g} in {csv_path}")
        return

    ml, mr, mt, mb = 76.0, 22.0, 52.0, 86.0
    x0, y0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb
    all_errors = [p["error"] for points in data.values() for p in points]
    all_qps = [p["qps"] for points in data.values() for p in points]
    min_error, max_error = 1e-4, 0.5
    xmin = max(min(all_errors), min_error)
    xmax = min(max(all_errors), max_error)
    if xmin >= xmax:
        xmin, xmax = min_error, max_error
    pad = max((math.log10(xmax) - math.log10(xmin)) * 0.08, 0.08)
    lx_min = max(math.log10(min_error), math.log10(xmin) - pad)
    lx_max = min(math.log10(max_error), math.log10(xmax) + pad)
    ytick_values, ytop = qps_ticks(max(all_qps))

    def sx(value: float) -> float:
        return x0 + (lx_max - math.log10(max(value, 1e-12))) / (lx_max - lx_min) * pw

    def sy(value: float) -> float:
        return y0 + ph - (value / QPS_SCALE) / ytop * ph

    out = [f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="white" stroke="#111" stroke-width="{font(PANEL_STROKE)}"/>']
    for tick in log_minor_ticks(10.0**lx_min, 10.0**lx_max):
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#d4d4d4" stroke-width="0.5" opacity="0.50"/>')
    for tick in log_ticks(10.0**lx_min, 10.0**lx_max):
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#cfcfcf" stroke-width="0.6" opacity="0.65"/>')
        out.append(log_tick_text(sx(tick), y0 + ph + 24, tick))
    for tick in ytick_values:
        out.append(f'<line x1="{x0:.2f}" y1="{y0 + ph - tick / ytop * ph:.2f}" x2="{x0+pw:.2f}" y2="{y0 + ph - tick / ytop * ph:.2f}" stroke="#d4d4d4" stroke-width="0.6" opacity="0.70"/>')
        out.append(f'<text x="{x0 - 6:.2f}" y="{y0 + ph - tick / ytop * ph + 5:.2f}" text-anchor="end" font-size="{font(TICK_FONT)}">{escape(fmt_axis(tick))}</text>')

    arm_order = ["hidann_index_diskann_search", "hidann_index_pipeann_search", "hidann_index_hidann_search"]
    for arm in arm_order:
        style = SEARCH_STYLE[arm]
        xy = [(sx(p["error"]), sy(p["qps"])) for p in data.get(arm, [])]
        if len(xy) >= 2:
            out.append(f'<path d="{line_path(xy)}" fill="none" stroke="{style["color"]}" stroke-width="{font(SERIES_STROKE)}"/>')
        for x, y in xy:
            out.append(marker(style["marker"], x, y, style["color"], POINT_MARKER_SIZE))

    present = [arm for arm in arm_order if data.get(arm)]
    out.append(legend([(SEARCH_STYLE[a]["label"], SEARCH_STYLE[a]["color"], SEARCH_STYLE[a]["marker"]) for a in present], x0 + 38, 25))
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 38:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700">1 - Recall@10</text>')
    out.append(f'<text x="24" y="{y0 + ph / 2:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700" transform="rotate(-90 24 {y0 + ph / 2:.2f})">QPS (&#215;10^4)</text>')
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 9:.2f}" text-anchor="middle" font-size="{font(PANEL_FONT)}" font-weight="700">{escape(dataset_label)}</text>')
    write_svg(
        plot_path(out_dir, dataset_key, "fig15", "search_same_hidann_index"),
        "\n".join(out),
        width,
        height,
    )


def draw_main_comparison(results_dir: Path, out_dir: Path, width: float, height: float, search_min_l: float, dataset_label: str, dataset_key: str) -> None:
    csv_path = results_dir / "search_full_system.csv"
    if not csv_path.is_file():
        print(f"skipping main comparison plot; missing {csv_path}")
        return
    rows = read_csv(csv_path)
    data: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        l_value = float(row["L"])
        if l_value < search_min_l:
            continue
        data[row["arm"]].append(
            {
                "L": l_value,
                "qps": float(row["qps"]),
                "recall": float(row["recall_at_10"]),
            }
        )
    for points in data.values():
        points.sort(key=lambda p: p["recall"])

    if not data:
        print(f"skipping main comparison plot; no rows with L >= {search_min_l:g} in {csv_path}")
        return

    ml, mr, mt, mb = 76.0, 22.0, 52.0, 86.0
    x0, y0 = ml, mt
    pw, ph = width - ml - mr, height - mt - mb
    all_recalls = [p["recall"] for points in data.values() for p in points]
    all_qps = [p["qps"] for points in data.values() for p in points]
    x_min = max(0.0, min(all_recalls) - 0.015)
    x_max = min(1.0, max(all_recalls) + 0.004)
    if x_max <= x_min:
        x_max = x_min + 0.01
    ytick_values, ytop = qps_ticks(max(all_qps))

    def sx(value: float) -> float:
        return x0 + (value - x_min) / (x_max - x_min) * pw

    def sy(value: float) -> float:
        return y0 + ph - (value / QPS_SCALE) / ytop * ph

    out = [f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{pw:.2f}" height="{ph:.2f}" fill="white" stroke="#111" stroke-width="{font(PANEL_STROKE)}"/>']
    for i in range(6):
        tick = x_min + (x_max - x_min) * i / 5
        out.append(f'<line x1="{sx(tick):.2f}" y1="{y0:.2f}" x2="{sx(tick):.2f}" y2="{y0+ph:.2f}" stroke="#d4d4d4" stroke-width="0.6" opacity="0.70"/>')
        out.append(f'<text x="{sx(tick):.2f}" y="{y0 + ph + 24:.2f}" text-anchor="middle" font-size="{font(TICK_FONT)}">{fmt_recall(tick)}</text>')
    for tick in ytick_values:
        y = y0 + ph - tick / ytop * ph
        out.append(f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x0+pw:.2f}" y2="{y:.2f}" stroke="#d4d4d4" stroke-width="0.6" opacity="0.70"/>')
        out.append(f'<text x="{x0 - 6:.2f}" y="{y + 5:.2f}" text-anchor="end" font-size="{font(TICK_FONT)}">{escape(fmt_axis(tick))}</text>')

    dynamic_styles: dict[str, dict[str, str]] = dict(SEARCH_STYLE)
    fallback = [
        {"label": "Run", "color": "#001F3F", "marker": "circle"},
        {"label": "Run 2", "color": "#CC0000", "marker": "square"},
        {"label": "Run 3", "color": "#556B2F", "marker": "star"},
    ]
    for idx, arm in enumerate(sorted(data)):
        dynamic_styles.setdefault(arm, fallback[idx % len(fallback)])

    for arm, style in dynamic_styles.items():
        xy = [(sx(p["recall"]), sy(p["qps"])) for p in data.get(arm, [])]
        if len(xy) >= 2:
            out.append(f'<path d="{line_path(xy)}" fill="none" stroke="{style["color"]}" stroke-width="{font(SERIES_STROKE)}"/>')
        for x, y in xy:
            out.append(marker(style["marker"], x, y, style["color"], POINT_MARKER_SIZE))

    out.append(legend([(dynamic_styles[a]["label"], dynamic_styles[a]["color"], dynamic_styles[a]["marker"]) for a in sorted(data)], x0 + 38, 25))
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 38:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700">Recall@10</text>')
    out.append(f'<text x="24" y="{y0 + ph / 2:.2f}" text-anchor="middle" font-size="{font(AXIS_FONT)}" font-weight="700" transform="rotate(-90 24 {y0 + ph / 2:.2f})">QPS (&#215;10^4)</text>')
    out.append(f'<text x="{x0 + pw / 2:.2f}" y="{height - 9:.2f}" text-anchor="middle" font-size="{font(PANEL_FONT)}" font-weight="700">{escape(dataset_label)}</text>')
    write_svg(
        plot_path(out_dir, dataset_key, "fig6", "main_comparison"),
        "\n".join(out),
        width,
        height,
    )


def main() -> int:
    args = parse_args()
    dataset_key = dataset_slug(args.dataset_slug or args.dataset_label)
    draw_quality(args.results_dir, args.out_dir, args.width, args.height, args.dataset_label, dataset_key)
    draw_main_comparison(args.results_dir, args.out_dir, args.width, args.height, args.search_min_l, args.dataset_label, dataset_key)
    draw_construction(args.results_dir, args.out_dir, args.width, args.height, args.dataset_label, dataset_key)
    draw_construction_effect(args.results_dir, args.out_dir, args.width, args.height, args.search_min_l, args.dataset_label, dataset_key)
    draw_same_index_search(args.results_dir, args.out_dir, args.width, args.height, args.search_min_l, args.dataset_label, dataset_key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

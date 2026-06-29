#!/usr/bin/env python3
"""Create a small deterministic float dataset and exact top-k ground truth."""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/toy"))
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--gt-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def write_fbin(path: Path, rows: list[list[float]]) -> None:
    if not rows:
        raise ValueError("empty matrix")
    dim = len(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<II", len(rows), dim))
        pack = struct.Struct("<" + "f" * dim)
        for row in rows:
            f.write(pack.pack(*row))


def write_truthset(path: Path, ids: list[list[int]], dists: list[list[float]]) -> None:
    if not ids:
        raise ValueError("empty truthset")
    nqueries = len(ids)
    k = len(ids[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<ii", nqueries, k))
        for row in ids:
            f.write(struct.pack("<" + "I" * k, *row))
        for row in dists:
            f.write(struct.pack("<" + "f" * k, *row))


def l2sq(a: list[float], b: list[float]) -> float:
    return sum((x - y) * (x - y) for x, y in zip(a, b))


def make_dataset(n: int, queries: int, dim: int, seed: int) -> tuple[list[list[float]], list[list[float]]]:
    rng = random.Random(seed)
    clusters = max(4, min(16, int(math.sqrt(n))))
    centers = [[rng.uniform(-1.0, 1.0) for _ in range(dim)] for _ in range(clusters)]

    base: list[list[float]] = []
    for i in range(n):
        center = centers[i % clusters]
        base.append([c + rng.gauss(0.0, 0.045) for c in center])

    query_rows: list[list[float]] = []
    for i in range(queries):
        source = base[(i * 37) % n]
        query_rows.append([x + rng.gauss(0.0, 0.015) for x in source])

    return base, query_rows


def exact_gt(base: list[list[float]], queries: list[list[float]], k: int) -> tuple[list[list[int]], list[list[float]]]:
    gt_ids: list[list[int]] = []
    gt_dists: list[list[float]] = []
    for q in queries:
        ranked = sorted(((l2sq(q, x), i) for i, x in enumerate(base)), key=lambda item: (item[0], item[1]))
        top = ranked[:k]
        gt_ids.append([i for _, i in top])
        gt_dists.append([d for d, _ in top])
    return gt_ids, gt_dists


def main() -> int:
    args = parse_args()
    if args.n < 512:
        raise SystemExit("--n must be at least 512 so PQ training has enough examples")
    if args.gt_k <= 0:
        raise SystemExit("--gt-k must be positive")
    if args.dim <= 0:
        raise SystemExit("--dim must be positive")

    base, queries = make_dataset(args.n, args.queries, args.dim, args.seed)
    ids, dists = exact_gt(base, queries, args.gt_k)

    write_fbin(args.out_dir / "base.fbin", base)
    write_fbin(args.out_dir / "query.fbin", queries)
    write_truthset(args.out_dir / f"gt{args.gt_k}.bin", ids, dists)

    print(f"base={args.out_dir / 'base.fbin'} n={args.n} dim={args.dim}")
    print(f"query={args.out_dir / 'query.fbin'} n={args.queries} dim={args.dim}")
    print(f"gt={args.out_dir / f'gt{args.gt_k}.bin'} k={args.gt_k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

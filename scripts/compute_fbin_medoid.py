#!/usr/bin/env python3
"""Compute the vector id nearest to the dataset centroid for an fbin file."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-file", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--out-file", type=Path, default=None)
    return parser.parse_args()


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise SystemExit(f"fbin header is too short: {path}")
    n, dim = struct.unpack("<II", header)
    expected = 8 + n * dim * 4
    got = path.stat().st_size
    if got != expected:
        raise SystemExit(f"unexpected fbin size for {path}: got {got}, expected {expected}")
    return n, dim


def row_batches(path: Path, n: int, dim: int, batch_size: int):
    row_bytes = dim * 4
    with path.open("rb") as handle:
        handle.seek(8)
        for start in range(0, n, batch_size):
            count = min(batch_size, n - start)
            buf = handle.read(count * row_bytes)
            if len(buf) != count * row_bytes:
                raise SystemExit(f"short read at row {start}: {path}")
            arr = np.frombuffer(buf, dtype=np.float32).reshape(count, dim)
            yield start, arr


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    n, dim = read_header(args.base_file)
    centroid = np.zeros(dim, dtype=np.float64)
    for _, arr in row_batches(args.base_file, n, dim, args.batch_size):
        centroid += arr.astype(np.float64, copy=False).sum(axis=0)
    centroid /= float(n)

    best_id = 0
    best_dist = np.inf
    centroid32 = centroid.astype(np.float32)
    for start, arr in row_batches(args.base_file, n, dim, args.batch_size):
        diff = arr - centroid32
        dists = np.einsum("ij,ij->i", diff, diff)
        local = int(np.argmin(dists))
        dist = float(dists[local])
        if dist < best_dist:
            best_dist = dist
            best_id = start + local

    text = f"{best_id}\n"
    if args.out_file is not None:
        args.out_file.parent.mkdir(parents=True, exist_ok=True)
        args.out_file.write_text(text)
    print(best_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

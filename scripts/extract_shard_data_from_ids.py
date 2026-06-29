#!/usr/bin/env python3
"""Stream a DiskANN fbin shard from a uint32 id-map file."""

from __future__ import annotations

import argparse
import os
import struct
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_fbin", type=Path)
    parser.add_argument("ids_uint32_bin", type=Path)
    parser.add_argument("out_fbin", type=Path)
    parser.add_argument("--chunk-mb", type=int, default=64)
    return parser.parse_args()


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise SystemExit(f"short header: {path}")
    return struct.unpack("<II", header)


def read_ids(path: Path) -> list[int]:
    n, dim = read_header(path)
    if dim != 1:
        raise SystemExit(f"expected id-map dim=1, found dim={dim}: {path}")
    with path.open("rb") as handle:
        handle.seek(8)
        data = handle.read(n * 4)
    if len(data) != n * 4:
        raise SystemExit(f"short id-map payload: {path}")
    return list(struct.unpack(f"<{n}I", data))


def main() -> int:
    args = parse_args()
    n, dim = read_header(args.base_fbin)
    ids = read_ids(args.ids_uint32_bin)
    if any(idx >= n for idx in ids):
        raise SystemExit(f"id-map contains an id outside base range 0..{n - 1}")

    positions: dict[int, list[int]] = defaultdict(list)
    for rank, idx in enumerate(ids):
        positions[idx].append(rank)

    row_size = dim * 4
    chunk_rows = max(1, (args.chunk_mb * 1024 * 1024) // row_size)
    out_tmp = Path(str(args.out_fbin) + ".tmp")
    args.out_fbin.parent.mkdir(parents=True, exist_ok=True)

    with args.base_fbin.open("rb") as src, out_tmp.open("wb") as out:
        src.seek(8)
        out.write(struct.pack("<II", len(ids), dim))
        if ids:
            out.seek(8 + len(ids) * row_size - 1)
            out.write(b"\0")

        base = 0
        remaining = len(positions)
        while base < n and remaining > 0:
            count = min(chunk_rows, n - base)
            buf = src.read(count * row_size)
            if len(buf) != count * row_size:
                raise SystemExit(f"short base read at row {base}: {args.base_fbin}")
            view = memoryview(buf)
            for local in range(count):
                idx = base + local
                ranks = positions.pop(idx, None)
                if ranks is None:
                    continue
                off = local * row_size
                row = view[off : off + row_size]
                for rank in ranks:
                    out.seek(8 + rank * row_size)
                    out.write(row)
                remaining -= 1
            base += count

    if positions:
        raise SystemExit(f"failed to extract {len(positions)} ids")
    os.replace(out_tmp, args.out_fbin)
    print(f"extracted_rows={len(ids)} dim={dim} output={args.out_fbin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

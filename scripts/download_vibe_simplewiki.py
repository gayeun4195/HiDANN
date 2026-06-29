#!/usr/bin/env python3
"""Download VIBE SimpleWiki and convert it to HiDANN binary inputs."""

from __future__ import annotations

import argparse
import hashlib
import struct
import urllib.request
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised on reviewer machines
    h5py = None


DEFAULT_URL = (
    "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/"
    "simplewiki-openai-3072-normalized.hdf5"
)
DEFAULT_SHA256 = "5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/simplewiki"))
    parser.add_argument("--hdf5", type=Path, default=None, help="Use an existing HDF5 file instead of downloading")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--sha256", default=DEFAULT_SHA256, help="Expected HDF5 SHA256; empty disables verification")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--gt-distance-scale",
        type=float,
        default=None,
        help="Scale HDF5 GT distances before writing. By default, normalized VIBE distances use scale 2.0.",
    )
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted files")
    return parser.parse_args()


def progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(blocks * block_size, total_size)
    pct = done * 100.0 / total_size
    pct_step = int(pct)
    if pct_step == getattr(progress, "last_pct_step", -1) and done < total_size:
        return
    progress.last_pct_step = pct_step
    print(f"\rdownloaded {done / (1024 ** 3):.2f} / {total_size / (1024 ** 3):.2f} GiB ({pct:5.1f}%)", end="")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"downloading {url}")
    print(f"to {dest}")
    progress.last_pct_step = -1
    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    tmp.replace(dest)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_dataset(h5, names: tuple[str, ...]):
    for name in names:
        if name in h5:
            obj = h5[name]
            if h5py is not None and isinstance(obj, h5py.Dataset):
                return obj
    available = ", ".join(sorted(h5.keys()))
    raise KeyError(f"none of {names} found in HDF5; available top-level keys: {available}")


def write_fbin(path: Path, dset, batch_size: int, force: bool) -> None:
    if path.exists() and not force:
        print(f"keeping existing {path}")
        return
    if dset.ndim != 2:
        raise ValueError(f"{dset.name}: expected rank-2 matrix, got shape {dset.shape}")
    n, dim = int(dset.shape[0]), int(dset.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<II", n, dim))
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            arr = np.asarray(dset[start:end], dtype=np.float32, order="C")
            f.write(arr.tobytes(order="C"))
    print(f"wrote {path} n={n} dim={dim}")


def write_gt(path: Path, ids_dset, dists_dset, batch_size: int, force: bool, distance_scale: float) -> None:
    if path.exists() and not force:
        print(f"keeping existing {path}")
        return
    if ids_dset.shape != dists_dset.shape or ids_dset.ndim != 2:
        raise ValueError(f"GT shape mismatch: ids={ids_dset.shape}, dists={dists_dset.shape}")
    nqueries, k = int(ids_dset.shape[0]), int(ids_dset.shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(struct.pack("<II", nqueries, k))
        for start in range(0, nqueries, batch_size):
            end = min(start + batch_size, nqueries)
            ids = np.asarray(ids_dset[start:end], dtype=np.uint32, order="C")
            f.write(ids.tobytes(order="C"))
        for start in range(0, nqueries, batch_size):
            end = min(start + batch_size, nqueries)
            dists = np.asarray(dists_dset[start:end], dtype=np.float32, order="C")
            if distance_scale != 1.0:
                dists = (dists * np.float32(distance_scale)).astype(np.float32, copy=False)
            f.write(dists.tobytes(order="C"))
    print(f"wrote {path} queries={nqueries} k={k} distance_scale={distance_scale:g}")


def gt_distance_scale(h5, requested_scale: float | None) -> float:
    if requested_scale is not None:
        return requested_scale
    distance_attr = h5.attrs.get("distance", "")
    if isinstance(distance_attr, bytes):
        distance_attr = distance_attr.decode("utf-8", errors="replace")
    if str(distance_attr).lower() == "normalized":
        # VIBE stores normalized distance as 1 - inner_product. The vectors are
        # unit-normalized, so squared L2 is 2 - 2 * inner_product.
        return 2.0
    return 1.0


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hdf5_path = args.hdf5 or (args.out_dir / "simplewiki-openai-3072-normalized.hdf5")
    if args.hdf5 is None and not hdf5_path.exists():
        download(args.url, hdf5_path)
    elif args.hdf5 is not None and not hdf5_path.exists():
        raise SystemExit(f"missing HDF5 file: {hdf5_path}")

    if args.sha256:
        got = sha256_file(hdf5_path)
        if got != args.sha256:
            raise SystemExit(f"SHA256 mismatch for {hdf5_path}: got {got}, expected {args.sha256}")
        print(f"verified SHA256 {got}")

    if args.download_only:
        return 0

    if h5py is None:
        raise SystemExit(
            "missing Python package: h5py\n"
            "Install it with `python3 -m pip install h5py` or your distribution package manager."
        )

    with h5py.File(hdf5_path, "r") as h5:
        base = require_dataset(h5, ("train", "base", "database"))
        query = require_dataset(h5, ("test", "query", "queries"))
        neighbors = require_dataset(h5, ("neighbors", "knns", "ids"))
        distances = require_dataset(h5, ("distances", "dists"))
        distance_scale = gt_distance_scale(h5, args.gt_distance_scale)

        write_fbin(args.out_dir / "simplewiki.fbin", base, args.batch_size, args.force)
        write_fbin(args.out_dir / "simplewiki_query.fbin", query, args.batch_size, args.force)
        write_gt(
            args.out_dir / f"simplewiki_gt{int(neighbors.shape[1])}",
            neighbors,
            distances,
            args.batch_size,
            args.force,
            distance_scale,
        )

    print("SimpleWiki dataset is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

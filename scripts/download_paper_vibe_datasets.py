#!/usr/bin/env python3
"""Download and convert the VIBE datasets used by the HiDANN paper."""

from __future__ import annotations

import argparse
import hashlib
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - exercised on reviewer machines
    h5py = None


HF_BASE = "https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main"


@dataclass(frozen=True)
class VibeDataset:
    key: str
    hdf5_name: str
    out_dir: str
    out_prefix: str
    sha256: str = ""


DATASETS: dict[str, VibeDataset] = {
    "simplewiki": VibeDataset(
        key="simplewiki",
        hdf5_name="simplewiki-openai-3072-normalized.hdf5",
        out_dir="simplewiki",
        out_prefix="simplewiki",
        sha256="5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3",
    ),
    "agnews": VibeDataset(
        key="agnews",
        hdf5_name="agnews-mxbai-1024-euclidean.hdf5",
        out_dir="agnews",
        out_prefix="agnews",
    ),
    "gooqa": VibeDataset(
        key="gooqa",
        hdf5_name="gooaq-distilroberta-768-normalized.hdf5",
        out_dir="gooaq",
        out_prefix="gooaq",
    ),
    "yahoo": VibeDataset(
        key="yahoo",
        hdf5_name="yahoo-minilm-384-normalized.hdf5",
        out_dir="yahoo",
        out_prefix="yahoo",
    ),
}

DEFAULT_DATASETS = ("simplewiki", "agnews", "gooqa", "yahoo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "datasets",
        nargs="*",
        default=list(DEFAULT_DATASETS),
        help=f"Dataset keys to download. Allowed: {', '.join(DATASETS)}",
    )
    parser.add_argument("--out-root", type=Path, default=Path("data"))
    parser.add_argument("--hdf5-root", type=Path, default=None, help="Optional directory for existing/downloaded HDF5 files")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing converted files")
    parser.add_argument("--skip-sha256", action="store_true", help="Skip built-in SHA256 checks")
    return parser.parse_args()


def normalize_keys(keys: list[str]) -> list[str]:
    selected: list[str] = []
    for key in keys:
        if key in {"all", "vibe-all"}:
            raise SystemExit(
                "This artifact only downloads VIBE datasets used in the paper. "
                f"Use explicit keys from: {', '.join(DATASETS)}"
            )
        if key not in DATASETS:
            raise SystemExit(f"unknown VIBE paper dataset: {key}; allowed keys: {', '.join(DATASETS)}")
        if key not in selected:
            selected.append(key)
    return selected


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


def gt_distance_scale(h5) -> float:
    distance_attr = h5.attrs.get("distance", "")
    if isinstance(distance_attr, bytes):
        distance_attr = distance_attr.decode("utf-8", errors="replace")
    if str(distance_attr).lower() == "normalized":
        return 2.0
    return 1.0


def prepare_dataset(spec: VibeDataset, args: argparse.Namespace) -> None:
    hdf5_root = args.hdf5_root or (args.out_root / "_hdf5")
    hdf5_path = hdf5_root / spec.hdf5_name
    if not hdf5_path.exists():
        download(f"{HF_BASE}/{spec.hdf5_name}", hdf5_path)

    if spec.sha256 and not args.skip_sha256:
        got = sha256_file(hdf5_path)
        if got != spec.sha256:
            raise SystemExit(f"SHA256 mismatch for {hdf5_path}: got {got}, expected {spec.sha256}")
        print(f"verified SHA256 {got}")

    if args.download_only:
        return

    if h5py is None:
        raise SystemExit("missing Python package: h5py; install with `python3 -m pip install h5py`")

    out_dir = args.out_root / spec.out_dir
    with h5py.File(hdf5_path, "r") as h5:
        base = require_dataset(h5, ("train", "base", "database"))
        query = require_dataset(h5, ("test", "query", "queries"))
        neighbors = require_dataset(h5, ("neighbors", "knns", "ids"))
        distances = require_dataset(h5, ("distances", "dists"))
        distance_scale = gt_distance_scale(h5)

        write_fbin(out_dir / f"{spec.out_prefix}.fbin", base, args.batch_size, args.force)
        write_fbin(out_dir / f"{spec.out_prefix}_query.fbin", query, args.batch_size, args.force)
        write_gt(
            out_dir / f"{spec.out_prefix}_gt{int(neighbors.shape[1])}",
            neighbors,
            distances,
            args.batch_size,
            args.force,
            distance_scale,
        )


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    for key in normalize_keys(args.datasets):
        print(f"\n== {key} ==")
        prepare_dataset(DATASETS[key], args)
    print("\nVIBE paper datasets are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

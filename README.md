# HiDANN

This repository contains the source code, dataset preparation scripts, and
execution scripts required to reproduce the HiDANN artifact experiments. It
provides scripts to compare **HiDANN** with **DiskANN**, **SOGAIC**, and
**PipeANN** on the paper datasets supported by the artifact downloader.

HiDANN builds on components from
[DiskANN](https://github.com/microsoft/DiskANN). It keeps DiskANN-compatible
index and binary data layouts where they are used by the construction and
search pipeline. The bundled baseline source snapshots are provided under
`baselines/`:

- `baselines/DiskANN`
- `baselines/SOGAIC`
- `baselines/PipeANN`

Baseline upstream basis and patch intent are documented in
`BASELINE_PROVENANCE.md` and `BASELINE_PATCH_NOTES.md`. License and attribution
notes are in `LICENSE` and `NOTICE.txt`.

## 1. Hardware and Software Requirements

- **OS**: Linux, Ubuntu 20.04/22.04 recommended.
- **Compiler**: GCC with C++17 support.
- **C++ dependencies**:
  - CMake
  - Boost program_options
  - Eigen3 and pybind11 headers
  - OpenMP
  - gperftools/tcmalloc
  - Linux AIO
  - Intel MKL, ILP64 runtime
- **Python dependencies**:
  - Python 3.8+
  - NumPy and h5py for the VIBE downloader/converter

On Ubuntu:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv \
  python3-dev pybind11-dev libeigen3-dev \
  libboost-program-options-dev libgoogle-perftools-dev \
  libaio-dev liburing-dev libmkl-full-dev
```

For Python:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install numpy h5py pybind11
```

Run the local dependency check with:

```bash
scripts/check_system.sh
```

The Ubuntu `pybind11-dev` and `libeigen3-dev` packages are the simplest setup.
If using a virtual environment or a custom Eigen install, the preflight and
PipeANN build also accept:

```bash
export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
export EIGEN3_INCLUDE_DIR=/path/containing/Eigen/Core
```

The search binaries use Linux AIO. If the host's `/proc/sys/fs/aio-max-nr` is
too small for the requested search thread count, the run scripts automatically
cap `SEARCH_THREADS` and report the cap. To keep 128 search threads on a fresh
Ubuntu machine, a typical setting is:

```bash
sudo sysctl fs.aio-max-nr=1048576
```

## 2. Setup and Build

Build HiDANN:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build build --target \
  our_construction our_pruning reverse_edge_logic create_disk_layout \
  evaluate_memory_index_quality ours_search -j $(nproc)
```

Build the bundled baselines:

```bash
cmake -S baselines/DiskANN -B baselines/DiskANN/build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build baselines/DiskANN/build --target \
  build_memory_index search_disk_index partition_with_ram_budget \
  extract_shard_data_from_ids merge_shards create_disk_layout \
  generate_pq gen_random_slice -j $(nproc)

cmake -S baselines/SOGAIC -B baselines/SOGAIC/build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build baselines/SOGAIC/build --target build_memory_index sogaic_partition_data merge_shards -j $(nproc)

cmake -S baselines/PipeANN -B baselines/PipeANN/build-aio \
  -DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON \
  ${pybind11_DIR:+-Dpybind11_DIR="${pybind11_DIR}"} \
  ${EIGEN3_INCLUDE_DIR:+-DEIGEN3_INCLUDE_DIR="${EIGEN3_INCLUDE_DIR}"}
cmake --build baselines/PipeANN/build-aio --target search_disk_index build_memory_index gen_random_slice -j $(nproc)
```

The convenience comparison script builds these targets automatically before a
full run.

## 3. Dataset Preparation

Download and convert VIBE SimpleWiki:

```bash
scripts/download_vibe_simplewiki.py --out-dir data/simplewiki
```

This creates:

- `data/simplewiki/simplewiki.fbin`
- `data/simplewiki/simplewiki_query.fbin`
- `data/simplewiki/simplewiki_gt100`

The dataset files are not committed to this repository. If the download fails,
place `simplewiki-openai-3072-normalized.hdf5` in the output directory and
rerun:

```bash
scripts/download_vibe_simplewiki.py \
  --hdf5 data/simplewiki/simplewiki-openai-3072-normalized.hdf5 \
  --out-dir data/simplewiki
```

To download the VIBE datasets used by the paper experiments:

```bash
scripts/download_paper_vibe_datasets.py simplewiki agnews gooqa yahoo
```

The script is restricted to the VIBE datasets used in the paper. Figure 7 also
uses MSTuring10M:

```bash
scripts/download_paper_msturing.sh msturing10M
```

The VIBE workflow uses the Hugging Face dataset
`vector-index-bench/vibe`. The MSTuring workflow uses the official
`big-ann-benchmarks` dataset loader and creates symlinks under
`data/msturing/msturing10M/`.

## 4. Running the Experiments

Small HiDANN smoke test:

```bash
scripts/run_artifact_pipeline.sh all
```

HiDANN-only dataset pipeline example:

```bash
scripts/run_dataset_pipeline.sh data/simplewiki runs/simplewiki
```

Integrated comparison on any supported prepared dataset:

```bash
scripts/run_paper_dataset_comparison.sh simplewiki all
scripts/run_paper_dataset_comparison.sh agnews all
scripts/run_paper_dataset_comparison.sh gooqa all
scripts/run_paper_dataset_comparison.sh yahoo all
scripts/run_paper_dataset_comparison.sh msturing10M all
```

Set `DOWNLOAD=1` to prepare the dataset before running:

```bash
DOWNLOAD=1 scripts/run_paper_dataset_comparison.sh yahoo all
```

Use `SEARCH_THREADS=<n>` to choose the search thread count independently from
the build/construction `THREADS` value. The scripts also pass `BUILD_DIR`
through to the Python construction driver, so custom build directories work
with the same wrappers.

The comparison wrapper runs:

- HiDANN exhaustive construction, sparse cross-partition probing construction,
  disk-layout generation, and warmed multi-`L` search.
- DiskANN construction through the common-PQ path:
  `generate_pq -> partition_with_ram_budget -> per-shard build_memory_index ->
  merge_shards -> create_disk_layout`.
- SOGAIC limited-memory construction.
- DiskANN search and PipeANN search on the DiskANN index.
- A B=1, single-common-entry memory-index quality evaluation for DiskANN,
  SOGAIC, and HiDANN.
- Fresh summary CSV generation and SVG plot generation under `runs/`.

Generated CSVs, SVGs, indexes, logs, and datasets are written under `runs/` and
`data/`, which are ignored by Git. The lower-level
`scripts/run_dataset_comparison.sh` entry point is used internally by the
dataset wrapper and can also be run directly with dataset environment
variables.

## 5. Reproducing Paper Figures

This artifact supports regenerating the following paper-style outputs. The
wrapper can be run on any downloadable dataset listed above.

| Paper figure | Artifact output |
|---|---|
| Figure 6: end-to-end QPS-Recall comparison | `runs/<dataset>_comparison_*/plots/fig6_main_comparison_<dataset>.svg` |
| Figure 7: construction-effect QPS-Recall comparison | `runs/<dataset>_comparison_*/plots/fig7_construction_effect_qps_<dataset>.svg` |
| Figure 9: construction quality comparison | `runs/<dataset>_comparison_*/plots/fig9_main_index_quality_<dataset>.svg` |
| Figure 15: same-HiDANN-index search comparison | `runs/<dataset>_comparison_*/plots/fig15_search_same_hidann_index_<dataset>.svg` |
| Table 3: construction time comparison | `runs/<dataset>_comparison_*/plots/table3_construction_time_<dataset>.svg` |

Figure 7 in the paper is drawn on Yahoo and MSTuring10M; the dataset wrapper can
run the same construction-effect protocol on those datasets after download.

End-to-end SimpleWiki reproduction:

```bash
# Step 1: Download dataset
scripts/download_vibe_simplewiki.py --out-dir data/simplewiki

# Step 2: Build and run HiDANN + baselines
scripts/run_paper_dataset_comparison.sh simplewiki all

# Step 3: Inspect generated summaries and plots
cat runs/simplewiki_comparison_*/summary/run_report.txt
ls runs/simplewiki_comparison_*/plots
```

The expected generated plots are:

- `runs/simplewiki_comparison_*/plots/fig6_main_comparison_simplewiki.svg`
- `runs/simplewiki_comparison_*/plots/table3_construction_time_simplewiki.svg`
- `runs/simplewiki_comparison_*/plots/fig7_construction_effect_qps_simplewiki.svg`
- `runs/simplewiki_comparison_*/plots/fig15_search_same_hidann_index_simplewiki.svg`
- `runs/simplewiki_comparison_*/plots/fig9_main_index_quality_simplewiki.svg`

The comparison report records construction time, full-system search,
same-HiDANN-index search, and construction-quality results. The same-index
search plot fixes the HiDANN disk layout and compares DiskANN, PipeANN, and
HiDANN search. The construction-quality stage computes one common medoid from
the base vectors and uses it as the shared entry point for every method with
beam width 1.

More detailed command breakdowns are in `docs/PIPELINES.md`.

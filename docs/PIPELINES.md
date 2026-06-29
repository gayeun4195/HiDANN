# Pipelines

All commands below run from the repository root. The convenience script
`scripts/run_artifact_pipeline.sh all` executes the same steps automatically.

## 1. Generate The Toy Dataset

```bash
python3 scripts/make_toy_dataset.py --out-dir data/toy --n 2048 --queries 100 --dim 32 --gt-k 10
```

This writes DiskANN-style binary files:

- `base.fbin`: `uint32 n`, `uint32 dim`, then row-major float32 vectors
- `query.fbin`: same format for query vectors
- `gt10.bin`: `int32 nqueries`, `int32 k`, then uint32 ids and float32 distances

## 2. VIBE and MSTuring Datasets

The toy dataset above is sufficient for the default smoke test. For artifact
comparison runs, the supported downloadable datasets are:

```text
simplewiki
agnews
gooqa
yahoo
msturing10M
```

To run a lightweight check, download VIBE SimpleWiki from Hugging Face and
convert it to the binary files consumed by HiDANN:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install numpy h5py
scripts/download_vibe_simplewiki.py --out-dir data/simplewiki
```

This writes:

- `data/simplewiki/simplewiki.fbin`
- `data/simplewiki/simplewiki_query.fbin`
- `data/simplewiki/simplewiki_gt100`

The downloader verifies the upstream HDF5 SHA256 by default. If downloading
from Hugging Face fails, place `simplewiki-openai-3072-normalized.hdf5` in the
output directory and rerun:

```bash
scripts/download_vibe_simplewiki.py \
  --hdf5 data/simplewiki/simplewiki-openai-3072-normalized.hdf5 \
  --out-dir data/simplewiki
```

The paper VIBE downloader is restricted to the VIBE datasets used by HiDANN:

```bash
scripts/download_paper_vibe_datasets.py simplewiki agnews gooqa yahoo
```

The MSTuring downloader is restricted to the MSTuring dataset used by HiDANN:

```bash
scripts/download_paper_msturing.sh msturing10M
```

## 3. Build HiDANN

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build build --target \
  our_construction our_pruning reverse_edge_logic create_disk_layout \
  evaluate_memory_index_quality ours_search -j 8
```

Use `-DPORTABLE=OFF` only when compiling on the same CPU class used for the
experiment; it enables `-march=native`.

## 4. Exhaustive Construction

```bash
HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
HIDANN_CONSTRUCTION_FADVISE=0 \
HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
python3 construction/run_sigma_pipeline.py \
  --data data/toy/base.fbin \
  --prefix runs/toy/full/construction/toy \
  --build-dir build \
  --P 4 --R 16 --L 20 --QD 8 --threads 8 \
  --pq-sampling-rate 1.0 \
  --memory-budget-bytes 1073741824 \
  --chunk-budget 268435456
```

The final graph is:

```text
runs/toy/full/construction/toy_P4_QD8_final.mem.index
```

## 5. Sparse Cross-Partition Probing Construction

Create an unordered cross-pair allowlist:

```bash
mkdir -p runs/toy/sparse
printf '0 1\n1 2\n2 3\n' > runs/toy/sparse/cross_pairs_P4.txt
```

Run the same construction pipeline with `--cross-pairs-file` to build a
sparse-probing index:

```bash
HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
HIDANN_CONSTRUCTION_FADVISE=0 \
HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
python3 construction/run_sigma_pipeline.py \
  --data data/toy/base.fbin \
  --prefix runs/toy/sparse/construction/toy \
  --build-dir build \
  --P 4 --R 16 --L 20 --QD 8 --threads 8 \
  --pq-sampling-rate 1.0 \
  --memory-budget-bytes 1073741824 \
  --chunk-budget 268435456 \
  --cross-pairs-file runs/toy/sparse/cross_pairs_P4.txt
```

The construction log reports both the total unordered cross-partition pairs and
the processed sparse-probing pairs. For `P=4`, exhaustive construction considers
6 unordered cross-partition pairs; the allowlist above processes 3.

## 6. Prepare Search Layout

```bash
mkdir -p runs/toy/full/layout
cp runs/toy/full/construction/toy_QD8.pq_compressed.bin runs/toy/full/layout/toy_QD8_pq_compressed.bin
cp runs/toy/full/construction/toy_QD8.pq_pivots.bin runs/toy/full/layout/toy_QD8_pq_pivots.bin
build/tools/create_disk_layout \
  float \
  data/toy/base.fbin \
  runs/toy/full/construction/toy_P4_QD8_final.mem.index \
  runs/toy/full/layout/toy_QD8
```

This writes `runs/toy/full/layout/toy_QD8_disk.index`.

## 7. Search

```bash
mkdir -p runs/toy/full/search
SIGMA_LAYOUT=COMBINED \
BATCH_REFINE_SIZE=16 \
build/search/ours_search \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix runs/toy/full/layout/toy_QD8 \
  --query_file data/toy/query.fbin \
  --gt_file data/toy/gt10.bin \
  --recall_at 10 \
  --search_list 20 \
  --beamwidth 2 \
  --num_threads 8 \
  --num_nodes_to_cache 0 \
  --result_path runs/toy/full/search/toy \
  --fail_if_recall_below 0.50
```

Repeat the layout and search steps with `runs/toy/sparse/...` to run HiDANN
search on the sparse-probing index.

## 8. Paper-Like SimpleWiki Check

The recommended dataset-level comparison entry point is:

```bash
DOWNLOAD=1 scripts/run_paper_dataset_comparison.sh simplewiki all
DOWNLOAD=1 scripts/run_paper_dataset_comparison.sh yahoo all
DOWNLOAD=1 scripts/run_paper_dataset_comparison.sh msturing10M all
```

The lower-level HiDANN-only dataset entry point can be run directly once the
dataset-specific environment variables are set. For example:

```bash
scripts/run_dataset_pipeline.sh data/simplewiki runs/simplewiki
```

It performs the full construction/search path, writes fresh summary CSVs under
`runs/simplewiki/summary/`, regenerates SVG plots under
`runs/simplewiki/plots/`, and leaves the generated artifacts outside the
Git-tracked submission files.

The commands below show the same steps explicitly for SimpleWiki. The dataset
wrapper fills the corresponding dataset path, prefix, `QD`, and budget settings
for each supported dataset.

```bash
BASE=data/simplewiki/simplewiki.fbin
QUERY=data/simplewiki/simplewiki_query.fbin
GT=data/simplewiki/simplewiki_gt100
RUN_ROOT=runs/simplewiki_paper_like

mkdir -p "${RUN_ROOT}/full/construction"
/usr/bin/time -v env \
  OMP_NUM_THREADS=128 \
  MKL_NUM_THREADS=128 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
  HIDANN_CONSTRUCTION_FADVISE=0 \
  HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
  python3 construction/run_sigma_pipeline.py \
    --data "${BASE}" \
    --prefix "${RUN_ROOT}/full/construction/simplewiki" \
    --build-dir build \
    --min-P 1 \
    --R 32 \
    --L 50 \
    --QD 512 \
    --threads 128 \
    --pq-sampling-rate 0.05 \
    --memory-budget-bytes 323277876 \
    --chunk-budget 161638938 \
    --partition-budget-fraction 0.95
```

To run the sparse cross-partition probing variant, create a deterministic
adjacent-pair allowlist for the auto-sized `P=24` case and pass it to the same
pipeline:

```bash
mkdir -p "${RUN_ROOT}/sparse"
PAIR_FILE="${RUN_ROOT}/sparse/cross_pairs_P24_adjacent.txt"
: > "${PAIR_FILE}"
for i in $(seq 0 22); do
  printf '%d %d\n' "${i}" "$((i + 1))" >> "${PAIR_FILE}"
done

mkdir -p "${RUN_ROOT}/sparse/construction"
/usr/bin/time -v env \
  OMP_NUM_THREADS=128 \
  MKL_NUM_THREADS=128 \
  OMP_PROC_BIND=close \
  OMP_PLACES=cores \
  HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
  HIDANN_CONSTRUCTION_FADVISE=0 \
  HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
  python3 construction/run_sigma_pipeline.py \
    --data "${BASE}" \
    --prefix "${RUN_ROOT}/sparse/construction/simplewiki_sparse_adjacent" \
    --build-dir build \
    --min-P 1 \
    --R 32 \
    --L 50 \
    --QD 512 \
    --threads 128 \
    --pq-sampling-rate 0.05 \
    --memory-budget-bytes 323277876 \
    --chunk-budget 161638938 \
    --partition-budget-fraction 0.95 \
    --cross-pairs-file "${PAIR_FILE}"
```

Prepare a disk layout by copying the PQ artifacts next to the disk index prefix:

```bash
mkdir -p "${RUN_ROOT}/full/layout"
cp "${RUN_ROOT}/full/construction/simplewiki_QD512.pq_compressed.bin" \
   "${RUN_ROOT}/full/layout/simplewiki_QD512_pq_compressed.bin"
cp "${RUN_ROOT}/full/construction/simplewiki_QD512.pq_pivots.bin" \
   "${RUN_ROOT}/full/layout/simplewiki_QD512_pq_pivots.bin"
build/tools/create_disk_layout \
  float \
  "${BASE}" \
  "${RUN_ROOT}/full/construction/simplewiki_P24_QD512_final.mem.index" \
  "${RUN_ROOT}/full/layout/simplewiki_QD512"
```

For search timing, run multiple `L` values in one process. The first `L` can
act like a warmup on some systems, so paper-like checks should compare the
warmed multi-`L` output rather than a single first query-list setting.

```bash
mkdir -p "${RUN_ROOT}/full/search/cache"
seq 0 260371 > "${RUN_ROOT}/full/search/cache/simplewiki_cache.txt"
SEARCH_THREADS="$(python3 scripts/preflight_aio.py --threads 128 --label simplewiki-search)"

CACHE_FILE="${RUN_ROOT}/full/search/cache/simplewiki_cache.txt" \
BATCH_REFINE_SIZE=32 \
EARLY_EXIT_STREAK=1 \
SIGMA_LAYOUT=COMBINED \
SIGMA_CACHE_LOOKUP_MODE=auto \
build/search/ours_search \
  --data_type float \
  --dist_fn l2 \
  --index_path_prefix "${RUN_ROOT}/full/layout/simplewiki_QD512" \
  --query_file "${QUERY}" \
  --gt_file "${GT}" \
  --recall_at 10 \
  --search_list 10 20 30 40 50 75 100 150 200 250 300 350 400 450 500 750 1000 \
  --beamwidth 8 \
  --num_threads "${SEARCH_THREADS}" \
  --num_nodes_to_cache 260372 \
  --result_path "${RUN_ROOT}/full/search/simplewiki" \
  --fail_if_recall_below 0
```

The manual run can then be summarized and plotted with:

```bash
python3 scripts/summarize_dataset_run.py --run-dir "${RUN_ROOT}"
python3 repro/figures/plot_dataset_reproduction.py \
  --results-dir "${RUN_ROOT}/summary" \
  --out-dir "${RUN_ROOT}/plots"
```

## 9. Baselines

Baseline source snapshots are available under `baselines/`:

- `baselines/DiskANN`
- `baselines/SOGAIC`
- `baselines/PipeANN`

Their provenance and patch intent are recorded in `BASELINE_PROVENANCE.md` and
`BASELINE_PATCH_NOTES.md`. The bundled scripts in those baseline directories are
helper scripts for the baseline snapshots and may need path overrides for a new
machine. For reviewer-facing reproduction, prefer the top-level dataset wrapper
and the documented parameters above.

## 10. Integrated Baseline and Quality Comparison

The reviewer-facing comparison entry point is:

```bash
USE_CGROUP=1 \
  scripts/run_paper_dataset_comparison.sh simplewiki all
```

This wrapper builds HiDANN and the bundled baseline targets, runs the HiDANN
construction/search path, runs the paper-style DiskANN and SOGAIC construction
baselines, runs DiskANN/PipeANN search baselines, reruns DiskANN/PipeANN search
on the HiDANN disk layout for the same-index diagnostic, runs the memory-index
quality stage, and then generates fresh CSVs and SVGs under:

```text
runs/<dataset>_comparison_<timestamp>/summary/
runs/<dataset>_comparison_<timestamp>/plots/
```

The default `all` mode writes the following primary plots:

```text
runs/<dataset>_comparison_<timestamp>/plots/fig6_main_comparison_<dataset>.svg
runs/<dataset>_comparison_<timestamp>/plots/fig7_construction_effect_qps_<dataset>.svg
runs/<dataset>_comparison_<timestamp>/plots/fig9_main_index_quality_<dataset>.svg
runs/<dataset>_comparison_<timestamp>/plots/fig15_search_same_hidann_index_<dataset>.svg
runs/<dataset>_comparison_<timestamp>/plots/table3_construction_time_<dataset>.svg
```

The full-system search plot uses Recall@10 on the x-axis. The
same-HiDANN-index search plot uses `1 - Recall@10` on a log-scaled x-axis. Both
search plots default to `--search-min-l 30`. The quality plot is controlled by
`QUALITY_L_VALUES` and is not cropped by this search-only option.

The Figure 7 construction-effect plot combines DiskANN-index + PipeANN search,
DiskANN-index + DiskANN search, and HiDANN-index + DiskANN search. The paper
Figure 7 uses Yahoo and MSTuring10M; the artifact downloader prepares those
datasets with:

```bash
scripts/download_paper_vibe_datasets.py yahoo
scripts/download_paper_msturing.sh msturing10M
```

The quality stage evaluates DiskANN, SOGAIC, and HiDANN memory graphs with
`beam_width=1` and `entry_protocol=single_common_entry`. By default, the common
entry is computed as the base vector nearest to the dataset centroid and then
shared across all methods. To rerun only that stage after the indexes already
exist:

```bash
RUN_DIR=runs/simplewiki_comparison USE_CGROUP=1 RUN_QUALITY=1 \
  scripts/run_paper_dataset_comparison.sh simplewiki \
  quality
```

The quality stage writes:

```text
runs/<dataset>_comparison_<timestamp>/summary/index_quality.csv
runs/<dataset>_comparison_<timestamp>/summary/quality_report.txt
runs/<dataset>_comparison_<timestamp>/plots/fig9_main_index_quality_<dataset>.svg
```

The evaluator maps the base vectors and reads the memory graph directly under
one common entry point. Use `QUALITY_CGROUP_LIMIT_BYTES` to adjust this memory
limit for larger datasets.

The generated comparison CSVs are for local inspection and figure generation;
they are not committed to the artifact repository.

The DiskANN construction baseline intentionally uses the common-PQ/manual-layout
path, not the one-shot compressed `build_disk_index`
path:

```text
generate_pq
partition_with_ram_budget
extract_shard_data_from_ids + build_memory_index for each shard
merge_shards
create_disk_layout
gen_random_slice
```

This matches the paper baseline, which searches a full-vector DiskANN disk
layout while keeping the resident PQ files next to the same prefix.

The wrapper uses `scripts/extract_shard_data_from_ids.py` for the shard data
materialization step. It streams the base file according to DiskANN's generated
uint32 id maps and writes the same fbin shard format without loading the full
base dataset into memory, so the preparation step also respects the artifact's
memory-capped execution.

For a faster partial rerun after a failed baseline step:

```bash
RUN_DIR=runs/simplewiki_comparison REUSE=1 scripts/run_paper_dataset_comparison.sh simplewiki diskann
RUN_DIR=runs/simplewiki_comparison REUSE=1 scripts/run_paper_dataset_comparison.sh simplewiki search
RUN_DIR=runs/simplewiki_comparison REUSE=1 scripts/run_paper_dataset_comparison.sh simplewiki pipeann
RUN_DIR=runs/simplewiki_comparison REUSE=1 scripts/run_paper_dataset_comparison.sh simplewiki summarize
RUN_DIR=runs/simplewiki_comparison REUSE=1 scripts/run_paper_dataset_comparison.sh simplewiki plots
```

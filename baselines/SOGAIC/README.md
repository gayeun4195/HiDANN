# SOGAIC Baseline Snapshot

This directory contains the SOGAIC construction baseline used in the HiDANN
paper experiments. It is a DiskANN-derived implementation with SOGAIC-specific
partitioning and limited-memory pipeline utilities added for the paper
evaluation.

This is not a general upstream DiskANN README. For provenance, see
`../../BASELINE_PROVENANCE.md`; for the purpose of the paper-campaign changes,
see `../../BASELINE_PATCH_NOTES.md`.

## What This Snapshot Adds

The SOGAIC implementation adds:

- `apps/utils/sogaic_partition_data.cpp`, the command-line partitioning driver.
- `src/sogaic_partition.cpp` and `include/sogaic_partition.h`, implementing
  overload-aware assignment, Gamma capacity checks, and memory-budget sizing.
- `scripts/run_sogaic_pipeline_limited.sh`, which runs partitioning, per-shard
  memory-index construction, merge, timing collection, and optional cleanup.

The implementation is intended for the construction comparison in the HiDANN
paper. Search quality for the resulting graph can be measured with the
DiskANN-compatible quality/search utilities used by the evaluation scripts.

## Build

Install the common HiDANN/DiskANN dependencies, including Intel MKL, Boost
program_options, gperftools/tcmalloc, and Linux AIO. Then build from this
directory:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build build -j 8
```

Use `-DPORTABLE=OFF` only when the binary will run on the same CPU class as the
build host.

## Example Construction Command

The following sketch mirrors the SimpleWiki paper setting. Adjust paths and
thread counts for the target machine:

```bash
scripts/run_sogaic_pipeline_limited.sh \
  data/simplewiki/simplewiki.fbin \
  runs/simplewiki_sogaic/simplewiki \
  --datatype float \
  --budget-ratio 0.1 \
  --memory-budget-bytes 323277876 \
  --partition-budget-fraction 0.95 \
  --chunk-budget 161638938 \
  --overlap 4.0 \
  --epsilon 1.8 \
  --gamma 1.0 \
  --R 32 \
  --L 50 \
  --threads 128
```

The script writes a log ending with a SOGAIC pipeline summary. The reported
stages are partitioning, per-shard index construction, merge, and total
pipeline time.

## Memory Accounting

SOGAIC uses the same logical memory-budget convention as the HiDANN artifact:
`--memory-budget-bytes` determines Gamma/capacity and shard feasibility. In the
formal campaign, commands were also run inside an OS cgroup cap with an
additional 512 MiB process/runtime margin. See `../../docs/PAPER_REPRODUCTION_NOTES.md`
for the exact convention.

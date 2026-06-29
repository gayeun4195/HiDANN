# Paper Reproduction Notes

This artifact is a compact functional package. It demonstrates exhaustive
construction, sparse cross-partition probing construction, disk-layout
preparation, and the HiDANN search path.

For larger runs, keep these settings in mind:

- Use `CMAKE_BUILD_TYPE=Release`.
- Use `-DPORTABLE=OFF` only when the binary will run on the same CPU class as
  the build host; otherwise keep `-DPORTABLE=ON`.
- Install Intel MKL and link the ILP64 interface. This artifact defines
  `MKL_ILP64` and links `mkl_intel_ilp64`, `mkl_intel_thread`, `mkl_core`,
  `iomp5`, `pthread`, `m`, and `dl`.
- The artifact repository is the reviewer-facing source package. Generated
  datasets, indexes, logs, CSVs, and plots are intentionally excluded from Git.

## Dataset Source

The optional SimpleWiki precheck uses the VIBE dataset file
`simplewiki-openai-3072-normalized.hdf5` from the Hugging Face dataset
`vector-index-bench/vibe`. The artifact downloader converts its `train`,
`test`, `neighbors`, and `distances` arrays into the HiDANN/DiskANN-style
`fbin` and ground-truth binary files.

The expected HDF5 SHA256 is:

```text
5b012e832cc6ffd72c248ce5e8ac12d34441c7692a4546e2a5672ebe2d4d4aa3
```

VIBE marks this file with `distance=normalized`, where distances are
`1 - inner_product` for unit-normalized vectors. The converter multiplies the
stored distances by 2 so the ground-truth distance file matches the squared-L2
distance convention used by the HiDANN/DiskANN binaries.

## Memory Accounting

The construction option `--memory-budget-bytes` is the logical/accounted
algorithm budget used to choose the number of partitions and to size the main
construction and pruning buffers. It is not, by itself, an OS-level RSS hard
limit. For budgeted construction, the artifact scripts can also run inside a
`systemd-run --user --scope` cgroup with:

```text
MemoryMax = memory_budget_bytes + 512 MiB
```

The 512 MiB margin covered common execution overhead such as the Python driver,
OpenMP/MKL runtime state, allocator retained heap, thread stacks, and transient
PQ-generation scratch buffers. For example, the SimpleWiki 10% construction
used `memory_budget_bytes=323277876` and `MemoryMax=860148788`.

When reproducing the budgeted setting on a machine with user-scoped systemd
available, run the construction command under the cgroup cap, for example:

```bash
systemd-run --user --scope \
  -p MemoryMax=860148788 \
  -p MemorySwapMax=0 \
  env OMP_NUM_THREADS=128 MKL_NUM_THREADS=128 OMP_PROC_BIND=close OMP_PLACES=cores \
  python3 construction/run_sigma_pipeline.py \
    --data data/simplewiki/simplewiki.fbin \
    --prefix runs/simplewiki_cgroup/full/construction/simplewiki \
    --build-dir build \
    --min-P 1 --R 32 --L 50 --QD 512 --threads 128 \
    --pq-sampling-rate 0.05 \
    --memory-budget-bytes 323277876 \
    --chunk-budget 161638938 \
    --partition-budget-fraction 0.95
```

This distinction matters most during the first pruning stage, where PQ pivots
and compressed codes are generated. The logical pruning budget accounts for the
resident PQ data plus the streaming batch budget, but PQ generation also holds a
temporary training sample and block buffers before those artifacts are reused by
the final pruning stage. Therefore an RSS peak above `--memory-budget-bytes`
does not by itself indicate that the logical HiDANN memory budget was violated;
reproduction runs should use both the logical budget arguments and the cgroup
execution cap when the host supports it.

The toy pipeline disables cache-drop/fadvise controls because they are not
needed for functional smoke testing. On a dedicated experiment server, those
controls can be enabled to mimic the paper timing environment.

## SimpleWiki HiDANN Pipeline

The command below runs the SimpleWiki construction path, an adjacent-pair sparse
cross-partition probing construction, layout generation, warmed multi-`L`
search, summary parsing, and SVG plot generation:

```bash
scripts/run_dataset_pipeline.sh data/simplewiki runs/simplewiki
```

The generated summaries live under `runs/simplewiki/summary/`, not in
the Git-tracked artifact.

## Baseline Sources

DiskANN, SOGAIC, and PipeANN source snapshots are bundled under `baselines/` for
review convenience. Their upstream basis and patch intent are documented in
`BASELINE_PROVENANCE.md` and `BASELINE_PATCH_NOTES.md`.

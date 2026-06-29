# Baseline Patch Notes

The bundled baselines are included for artifact reproduction. The changes
relative to the original upstream projects are compatibility, accounting, and
robustness changes needed to run the same evaluation protocol. They are not
intended to make any baseline slower or less accurate.

## DiskANN

The DiskANN baseline is derived from upstream DiskANN commit `a26f824`, with
the artifact compatibility changes summarized below.

The artifact changes do the following:

- Add utilities and build targets used by the construction pipeline, including
  shard extraction from id maps and standalone PQ generation.
- Add cache-drop controls for large sequential readers/writers. These controls
  make repeated construction runs less sensitive to leftover page cache on a
  shared server; they do not alter graph construction logic.
- Validate actual shard sizes against the memory budget after full assignment,
  retrying with more partitions when sampled estimates are too optimistic. This
  is a memory-budget correctness check.
- Preserve the OPQ/PQ selection when generating compressed vectors from pivots.
- Return errors when merged Vamana construction fails instead of continuing with
  incomplete outputs.
- Add an explicit `--saturate_graph` switch to `build_memory_index`. The default
  remains disabled, matching the paper baseline setting unless the caller opts
  in.
- Reserve cache hash tables before loading many cached nodes and fix merge
  output for missing leading ids. These are correctness/robustness fixes for the
  experiment inputs.
- Remove a configure-time debug print that listed every source file collected
  for the optional clang-format targets.

For construction-time comparisons, the artifact reports the paper's
`DiskANN common-PQ` time. This means DiskANN's merged-Vamana construction time
plus the same PQ-generation time charged to HiDANN, so the plotted comparison is
not advantaged or penalized by using different PQ accounting.

## SOGAIC

SOGAIC is the HiDANN paper's DiskANN-derived SOGAIC implementation, not a
verbatim upstream DiskANN release.

The SOGAIC changes add:

- A SOGAIC partitioning utility and limited-memory pipeline script.
- Gamma/capacity sizing from the same logical memory budget used by the paper
  evaluation.
- Assignment checks that fail if a point is left unassigned or a shard exceeds
  the selected Gamma capacity.
- Explicit memory-budget, chunk-budget, and temporary-file cleanup controls.
- Structured stage timing for partitioning, per-shard index construction, merge,
  and total pipeline time.
- Remove a configure-time debug print that listed every source file collected
  for the optional clang-format targets.

The purpose is to make SOGAIC runnable under the same memory-budget and timing
protocol as HiDANN and DiskANN. The code does not add artificial delays or
change reported quality metrics after the fact.

## PipeANN

The PipeANN baseline is derived from the PipeANN implementation used by the
paper, with the artifact compatibility changes summarized below.

The artifact changes do the following:

- Size neighbor scratch buffers from the actual PQ chunk count when that count
  exceeds the historical fixed 256-byte assumption. This avoids under-allocating
  search scratch space for the high-dimensional SimpleWiki/HiDANN layouts.
- Add runtime checks around `io_uring` queue initialization, SQE allocation,
  CQE polling, and read/write completion. These make I/O failures explicit
  instead of causing silent undefined behavior.
- Support the Linux AIO build path used by this artifact environment. The
  vendored `io_uring` headers in the snapshot are incomplete on some machines;
  the AIO build keeps the same search executable interface while avoiding that
  environment-specific build failure.
- Fix the AIO request metadata assignment in
  `src/utils/linux_aligned_file_reader.cpp` by writing through the vector data
  pointer. This is a compile-time compatibility fix for the intended per-request
  user-data pointer, not a change to PipeANN's search policy or a performance
  throttling change.
- Accept an explicit Eigen include directory during CMake configure so the
  artifact can build with either system Eigen or a user-provided include path.

These are robustness and input-compatibility changes for the search baseline.
They are not algorithmic changes intended to degrade PipeANN performance.

## Licenses

Each bundled baseline keeps its own `LICENSE` and `NOTICE.txt` files. The
top-level `NOTICE.txt` summarizes the DiskANN/NSG attribution for HiDANN itself;
baseline-specific attribution remains in the corresponding baseline directory.

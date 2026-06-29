# Construction Components

This directory contains the HiDANN construction executables used by the
artifact pipeline:

- `our_construction`: partition-local graph build plus self/cross pair
  candidate generation
- `our_pruning`: PQ-based pruning pass
- `reverse_edge_logic`: reverse-edge candidate generation
- `run_sigma_pipeline.py`: end-to-end construction driver

The artifact driver supports exhaustive construction by default. Sparse
cross-partition probing uses `run_sigma_pipeline.py --cross-pairs-file PATH`,
where `PATH` contains unordered partition pairs such as:

```text
0 1
1 2
2 3
```

See `docs/PIPELINES.md` for runnable commands.

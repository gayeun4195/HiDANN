# Bundled Baselines

This directory contains source snapshots for the baselines used in the HiDANN
paper artifact:

- `DiskANN/`: DiskANN construction/search baseline snapshot.
- `SOGAIC/`: HiDANN paper's DiskANN-derived SOGAIC construction baseline.
- `PipeANN/`: PipeANN search baseline snapshot.

The snapshots are included as plain files, not nested Git repositories. See
`../BASELINE_PROVENANCE.md` for exact commits and `../BASELINE_PATCH_NOTES.md`
for the purpose of the paper-campaign modifications.

## Build Sketch

The baselines inherit their upstream build systems. On Ubuntu, install the
dependencies listed in `../docs/DEPENDENCIES.md`, then configure each baseline
from its own build directory:

```bash
cmake -S baselines/DiskANN -B baselines/DiskANN/build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build baselines/DiskANN/build --target \
  build_memory_index search_disk_index partition_with_ram_budget \
  extract_shard_data_from_ids merge_shards create_disk_layout \
  generate_pq gen_random_slice -j 8

cmake -S baselines/SOGAIC -B baselines/SOGAIC/build -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
cmake --build baselines/SOGAIC/build --target \
  build_memory_index sogaic_partition_data merge_shards -j 8

cmake -S baselines/PipeANN -B baselines/PipeANN/build-aio -DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON
cmake --build baselines/PipeANN/build-aio --target \
  search_disk_index build_memory_index gen_random_slice -j 8
```

Paper-scale runs used Release builds, Intel MKL, and 128 threads. The exact
construction/search commands are documented in `../docs/PIPELINES.md`. The
top-level `scripts/run_simplewiki_comparison.sh` wrapper performs the baseline
builds and uses the paper's common-PQ DiskANN construction path for SimpleWiki.

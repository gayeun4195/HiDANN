# Search Components

This directory contains the HiDANN search executable used by the artifact:

- `ours_search`: searches a combined disk layout created from a HiDANN
  `.mem.index` graph plus HiDANN PQ files
- `ours_cache_gen`: optional cache-list helper, not needed by the default smoke
  pipeline

The artifact smoke uses:

```bash
SIGMA_LAYOUT=COMBINED build/search/ours_search ...
```

See `docs/PIPELINES.md` for the complete search command.

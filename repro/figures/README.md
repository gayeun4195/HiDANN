# Reproduction Plot Inputs

This directory contains a dependency-light SVG plotter for artifact run outputs.
It does not ship paper-result CSVs.

The run scripts write fresh CSV summaries under the selected run directory:

- `summary/construction_time.csv`: measured construction time and memory.
- `summary/search_full_system.csv`: warmed multi-`L` HiDANN search results.
- `summary/search_same_hidann_index.csv`: optional same-HiDANN-index DiskANN,
  PipeANN, and HiDANN search rows.
- `summary/index_quality.csv`: optional B=1 memory-index quality rows produced
  by `scripts/run_paper_dataset_comparison.sh ... quality`.
- `summary/run_report.txt`: compact run summary for quick inspection.
- `summary/pipeline_status.json`: machine-readable status for command failures
  and missing required outputs.

Regenerate SVG plots from a run summary with:

```bash
python3 repro/figures/plot_dataset_reproduction.py \
  --results-dir runs/<dataset>_comparison_<timestamp>/summary \
  --out-dir runs/<dataset>_comparison_<timestamp>/plots \
  --dataset-label <DatasetLabel> \
  --dataset-slug <dataset>
```

The Recall@10/QPS plots default to the paper search-figure crop
`--search-min-l 30`. The construction-quality plot is not cropped by this
option; it uses all rows present in `index_quality.csv`.

Depending on which CSVs are present, the plotter writes:

- `plots/fig6_main_comparison_<dataset>.svg`
- `plots/fig7_construction_effect_qps_<dataset>.svg`, only when both full-system and
  same-HiDANN-index search CSVs are present.
- `plots/fig15_search_same_hidann_index_<dataset>.svg`, only when
  `search_same_hidann_index.csv` is present.
- `plots/fig9_main_index_quality_<dataset>.svg`, only when `index_quality.csv` is present.
- `plots/table3_construction_time_<dataset>.svg`

The full-system search plot uses Recall@10 on the x-axis. The same-HiDANN-index
search plot follows the paper's same-index search convention: `1 - Recall@10`
on a log-scaled x-axis and QPS on the y-axis. The quality plot follows the
paper convention for graph-quality figures: `1 - Recall@10` on the x-axis,
`Average hop #` on the y-axis, both in log scale. The expected quality input
uses `entry_protocol=single_common_entry` and `beam_width=1`.
The comparison wrapper computes the shared entry point as the base vector
nearest to the dataset centroid, then reuses that one entry for every method.

The dataset comparison wrapper regenerates these plots after running the
pipeline:

```bash
scripts/run_paper_dataset_comparison.sh <dataset> all
```

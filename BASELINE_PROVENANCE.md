# Baseline Provenance

This artifact bundles baseline source snapshots as plain files under
`baselines/`. They are not nested Git repositories, so reviewers can build and
run the artifact from one repository.

| Component | Bundled path | Upstream basis |
|---|---|---|
| DiskANN baseline | `baselines/DiskANN` | Microsoft DiskANN, commit `a26f824` |
| SOGAIC baseline | `baselines/SOGAIC` | DiskANN-derived SOGAIC implementation used by HiDANN |
| PipeANN baseline | `baselines/PipeANN` | PipeANN implementation used by HiDANN, commit `269485e` |

The bundled snapshots include the small compatibility, accounting, and
robustness changes needed by the artifact scripts. Those changes are summarized
in `BASELINE_PATCH_NOTES.md`.

The HiDANN artifact source keeps the construction/search pipeline, dataset
conversion scripts, baseline wrappers, and run scripts needed to reproduce the
artifact figures.

## Import Method

The baseline directories were imported from tracked source trees only, using
the equivalent of:

```bash
git archive <baseline-source-commit> | tar -x -C baselines/<BaselineName>
```

Generated data, build directories, logs, and experiment outputs are excluded by
the top-level `.gitignore`.

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-simplewiki}"
MODE="${2:-all}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/${DATASET}_comparison_$(date -u +%Y%m%d_%H%M%S)}"
DOWNLOAD="${DOWNLOAD:-0}"
R="${R:-32}"
L="${L:-50}"
THREADS="${THREADS:-128}"
SEARCH_THREADS="${SEARCH_THREADS:-${THREADS}}"

usage() {
  cat <<EOF
Usage: $0 [simplewiki|agnews|gooqa|yahoo|msturing10M] [mode]

Set DOWNLOAD=1 to prepare the dataset before running. Mode is forwarded to
scripts/run_dataset_comparison.sh; common values are all, build, hidann,
diskann, sogaic, search, pipeann, same_search, quality, summarize, plots.
EOF
}

configure_dataset() {
  case "${DATASET}" in
    simplewiki)
      DATASET_NAME="simplewiki"
      DATASET_PREFIX="simplewiki"
      DATASET_LABEL="SimpleWiki"
      DATA_DIR="${DATA_DIR:-${ROOT}/data/simplewiki}"
      BASE_FILE="simplewiki.fbin"
      QUERY_FILE="simplewiki_query.fbin"
      GT_FILE="simplewiki_gt100"
      QD="${QD:-512}"
      ;;
    agnews)
      DATASET_NAME="agnews"
      DATASET_PREFIX="agnews"
      DATASET_LABEL="AGNews"
      DATA_DIR="${DATA_DIR:-${ROOT}/data/agnews}"
      BASE_FILE="agnews.fbin"
      QUERY_FILE="agnews_query.fbin"
      GT_FILE="agnews_gt100"
      QD="${QD:-256}"
      ;;
    gooqa)
      DATASET_NAME="gooqa"
      DATASET_PREFIX="gooaq"
      DATASET_LABEL="GooAQ"
      DATA_DIR="${DATA_DIR:-${ROOT}/data/gooaq}"
      BASE_FILE="gooaq.fbin"
      QUERY_FILE="gooaq_query.fbin"
      GT_FILE="gooaq_gt100"
      QD="${QD:-192}"
      ;;
    yahoo)
      DATASET_NAME="yahoo"
      DATASET_PREFIX="yahoo"
      DATASET_LABEL="Yahoo"
      DATA_DIR="${DATA_DIR:-${ROOT}/data/yahoo}"
      BASE_FILE="yahoo.fbin"
      QUERY_FILE="yahoo_query.fbin"
      GT_FILE="yahoo_gt100"
      QD="${QD:-96}"
      ;;
    msturing10M)
      DATASET_NAME="msturing10M"
      DATASET_PREFIX="msturing10M"
      DATASET_LABEL="MSTuring10M"
      DATA_DIR="${DATA_DIR:-${ROOT}/data/msturing/msturing10M}"
      BASE_FILE="msturing10M.fbin"
      QUERY_FILE="query100K.fbin"
      GT_FILE="msturing10M_gt100"
      QD="${QD:-25}"
      ;;
    -h|--help|help)
      usage
      exit 0
      ;;
    *)
      echo "unsupported dataset: ${DATASET}" >&2
      usage >&2
      exit 2
      ;;
  esac
}

download_dataset() {
  if [[ "${DOWNLOAD}" != "1" ]]; then
    return
  fi
  case "${DATASET}" in
    simplewiki|agnews|gooqa|yahoo)
      "${ROOT}/scripts/download_paper_vibe_datasets.py" "${DATASET}" --out-root "${ROOT}/data"
      ;;
    msturing10M)
      OUT_ROOT="${ROOT}/data/msturing" "${ROOT}/scripts/download_paper_msturing.sh" msturing10M
      ;;
  esac
}

require_dataset_files() {
  for file in "${DATA_DIR}/${BASE_FILE}" "${DATA_DIR}/${QUERY_FILE}" "${DATA_DIR}/${GT_FILE}"; do
    if [[ ! -f "${file}" ]]; then
      echo "missing dataset file: ${file}" >&2
      echo "prepare it with DOWNLOAD=1 $0 ${DATASET} ${MODE}" >&2
      exit 1
    fi
  done
}

fill_budget_defaults() {
  read -r auto_memory auto_chunk auto_cgroup auto_search auto_quality auto_pipeann auto_build_gb auto_partition_chunk auto_cache < <(
    python3 - "${DATA_DIR}/${BASE_FILE}" "${R}" <<'PY'
import math
import os
import struct
import sys

path = sys.argv[1]
r = int(sys.argv[2])
with open(path, "rb") as handle:
    n, dim = struct.unpack("<II", handle.read(8))
file_size = os.path.getsize(path)
index_size = file_size + n * r * 4
memory = max(1, int(index_size * 0.10))
chunk = max(1, memory // 2)
cgroup = memory + 512 * 1024 * 1024
search = max(cgroup, memory + 1280 * 1024 * 1024)
quality = max(search, file_size + 1024 * 1024 * 1024)
pipeann = max(search, memory + 1400 * 1024 * 1024)
build_gb = memory / (1024 ** 3)
cache_nodes = max(1, min(n, int(math.ceil(n * 0.0585))))
print(memory, chunk, cgroup, search, quality, pipeann, f"{build_gb:.6f}", chunk, cache_nodes)
PY
  )

  MEMORY_BUDGET_BYTES="${MEMORY_BUDGET_BYTES:-${auto_memory}}"
  CHUNK_BUDGET_BYTES="${CHUNK_BUDGET_BYTES:-${auto_chunk}}"
  CGROUP_LIMIT_BYTES="${CGROUP_LIMIT_BYTES:-${auto_cgroup}}"
  SEARCH_CGROUP_LIMIT_BYTES="${SEARCH_CGROUP_LIMIT_BYTES:-${auto_search}}"
  QUALITY_CGROUP_LIMIT_BYTES="${QUALITY_CGROUP_LIMIT_BYTES:-${auto_quality}}"
  PIPEANN_CGROUP_LIMIT_BYTES="${PIPEANN_CGROUP_LIMIT_BYTES:-${auto_pipeann}}"
  BUILD_DRAM_GB="${BUILD_DRAM_GB:-${auto_build_gb}}"
  DISKANN_PARTITION_CHUNK_BYTES="${DISKANN_PARTITION_CHUNK_BYTES:-${auto_partition_chunk}}"
  DISKANN_CACHE_NODES="${DISKANN_CACHE_NODES:-${auto_cache}}"
}

configure_dataset
download_dataset

if [[ "${MODE}" != "build" ]]; then
  require_dataset_files
  fill_budget_defaults
fi

export DATASET_NAME DATASET_PREFIX DATASET_LABEL BASE_FILE QUERY_FILE GT_FILE
export QD R L THREADS SEARCH_THREADS
export MEMORY_BUDGET_BYTES CHUNK_BUDGET_BYTES CGROUP_LIMIT_BYTES
export SEARCH_CGROUP_LIMIT_BYTES QUALITY_CGROUP_LIMIT_BYTES PIPEANN_CGROUP_LIMIT_BYTES
export BUILD_DRAM_GB DISKANN_PARTITION_CHUNK_BYTES DISKANN_CACHE_NODES

echo "dataset=${DATASET_NAME}"
echo "data_dir=${DATA_DIR}"
echo "run_dir=${RUN_DIR}"
echo "QD=${QD} memory_budget_bytes=${MEMORY_BUDGET_BYTES:-auto} search_threads=${SEARCH_THREADS}"

"${ROOT}/scripts/run_dataset_comparison.sh" "${DATA_DIR}" "${RUN_DIR}" "${MODE}"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_NAME="${DATASET_NAME:-simplewiki}"
DATASET_PREFIX="${DATASET_PREFIX:-${DATASET_NAME}}"
DATASET_LABEL="${DATASET_LABEL:-SimpleWiki}"
BASE_FILE="${BASE_FILE:-${DATASET_PREFIX}.fbin}"
QUERY_FILE="${QUERY_FILE:-${DATASET_PREFIX}_query.fbin}"
GT_FILE="${GT_FILE:-${DATASET_PREFIX}_gt100}"
DATA_DIR="${1:-${ROOT}/data/${DATASET_NAME}}"
RUN_DIR="${2:-${ROOT}/runs/${DATASET_NAME}_pipeline_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${3:-all}"

BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
THREADS="${THREADS:-128}"
SEARCH_THREADS="${SEARCH_THREADS:-${THREADS}}"
R="${R:-32}"
L="${L:-50}"
QD="${QD:-512}"
BEAM_WIDTH="${BEAM_WIDTH:-8}"
MEMORY_BUDGET_BYTES="${MEMORY_BUDGET_BYTES:-323277876}"
CHUNK_BUDGET_BYTES="${CHUNK_BUDGET_BYTES:-161638938}"
PARTITION_BUDGET_FRACTION="${PARTITION_BUDGET_FRACTION:-0.95}"
CGROUP_LIMIT_BYTES="${CGROUP_LIMIT_BYTES:-860148788}"
SEARCH_CGROUP_LIMIT_BYTES="${SEARCH_CGROUP_LIMIT_BYTES:-1705667336}"
USE_CGROUP="${USE_CGROUP:-1}"
REUSE="${REUSE:-0}"
RUN_SPARSE="${RUN_SPARSE:-1}"
L_VALUES=(${L_VALUES:-10 20 30 40 50 75 100 150 200 250 300 350 400 450 500 750 1000})

BASE="${DATA_DIR}/${BASE_FILE}"
QUERY="${DATA_DIR}/${QUERY_FILE}"
GT="${DATA_DIR}/${GT_FILE}"

usage() {
  cat <<EOF
Usage: $0 [data_dir] [run_dir] [all|build|construction|sparse|layout|search|summarize|plots]

Environment overrides:
  THREADS=${THREADS} R=${R} L=${L} QD=${QD} BEAM_WIDTH=${BEAM_WIDTH}
  MEMORY_BUDGET_BYTES=${MEMORY_BUDGET_BYTES}
  CHUNK_BUDGET_BYTES=${CHUNK_BUDGET_BYTES}
  PARTITION_BUDGET_FRACTION=${PARTITION_BUDGET_FRACTION}
  CGROUP_LIMIT_BYTES=${CGROUP_LIMIT_BYTES}
  SEARCH_CGROUP_LIMIT_BYTES=${SEARCH_CGROUP_LIMIT_BYTES}
  USE_CGROUP=${USE_CGROUP} REUSE=${REUSE} RUN_SPARSE=${RUN_SPARSE}
  SEARCH_THREADS=${SEARCH_THREADS}
EOF
}

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

require_data() {
  for path in "${BASE}" "${QUERY}" "${GT}"; do
    if [[ ! -f "${path}" ]]; then
      echo "missing ${path}" >&2
      echo "prepare ${DATASET_LABEL} under ${DATA_DIR}" >&2
      exit 1
    fi
  done
}

read_fbin_n() {
  python3 - "$1" <<'PY'
import struct
import sys
with open(sys.argv[1], "rb") as handle:
    print(struct.unpack("<I", handle.read(4))[0])
PY
}

build_artifact() {
  run_cmd cmake -S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
  run_cmd cmake --build "${BUILD_DIR}" --target our_construction our_pruning reverse_edge_logic create_disk_layout ours_search -j "${THREADS}"
}

run_scoped_time() {
  local memory_limit="$1"
  local unit_name="$2"
  shift 2

  if [[ "${USE_CGROUP}" == "1" ]] && command -v systemd-run >/dev/null 2>&1; then
    systemd-run --user --scope --unit="${unit_name}" \
      -p MemoryMax="${memory_limit}" \
      -p MemorySwapMax=0 \
      /usr/bin/time -v "$@"
  else
    /usr/bin/time -v "$@"
  fi
}

aio_checked_threads() {
  python3 "${ROOT}/scripts/preflight_aio.py" --threads "$1" --label "$2"
}

run_full_construction() {
  require_data
  local out_dir="${RUN_DIR}/full/construction"
  local prefix="${out_dir}/${DATASET_PREFIX}"
  local outer_log="${out_dir}/${DATASET_PREFIX}_construction_outer.log"
  mkdir -p "${out_dir}"

  if [[ "${REUSE}" == "1" ]] && compgen -G "${prefix}_P*_QD${QD}_final.mem.index" >/dev/null; then
    echo "reusing existing full construction in ${out_dir}"
    return
  fi

  HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
  HIDANN_CONSTRUCTION_FADVISE=0 \
  HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
  run_scoped_time "${CGROUP_LIMIT_BYTES}" "hidann_${DATASET_NAME}_construction_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      python3 "${ROOT}/construction/run_sigma_pipeline.py" \
        --data "${BASE}" \
        --prefix "${prefix}" \
        --min-P 1 \
        --R "${R}" \
        --L "${L}" \
        --QD "${QD}" \
        --threads "${THREADS}" \
        --build-dir "${BUILD_DIR}" \
        --pq-sampling-rate 0.05 \
        --memory-budget-bytes "${MEMORY_BUDGET_BYTES}" \
        --chunk-budget "${CHUNK_BUDGET_BYTES}" \
        --partition-budget-fraction "${PARTITION_BUDGET_FRACTION}" \
    2>&1 | tee "${outer_log}"
}

detect_full_graph() {
  local graphs=("${RUN_DIR}/full/construction"/"${DATASET_PREFIX}"_P*_QD${QD}_final.mem.index)
  if [[ ! -e "${graphs[0]}" ]]; then
    echo "missing full final graph under ${RUN_DIR}/full/construction" >&2
    exit 1
  fi
  echo "${graphs[0]}"
}

detect_full_p() {
  local graph
  graph="$(detect_full_graph)"
  basename "${graph}" | sed -E "s/${DATASET_PREFIX}_P([0-9]+)_QD${QD}_final\\.mem\\.index/\\1/"
}

write_sparse_pairs() {
  local pair_file="${RUN_DIR}/sparse/cross_pairs_P$(detect_full_p)_adjacent.txt"
  mkdir -p "$(dirname "${pair_file}")"
  : > "${pair_file}"
  local p
  p="$(detect_full_p)"
  for ((i = 0; i + 1 < p; ++i)); do
    printf '%d %d\n' "${i}" "$((i + 1))" >> "${pair_file}"
  done
  echo "${pair_file}"
}

run_sparse_construction() {
  if [[ "${RUN_SPARSE}" != "1" ]]; then
    echo "RUN_SPARSE=${RUN_SPARSE}; skipping sparse adjacent-pair construction"
    return
  fi
  require_data
  local out_dir="${RUN_DIR}/sparse/construction"
  local prefix="${out_dir}/${DATASET_PREFIX}_sparse_adjacent"
  local outer_log="${out_dir}/${DATASET_PREFIX}_sparse_construction_outer.log"
  mkdir -p "${out_dir}"

  if [[ "${REUSE}" == "1" ]] && compgen -G "${prefix}_P*_QD${QD}_final.mem.index" >/dev/null; then
    echo "reusing existing sparse construction in ${out_dir}"
    return
  fi

  local pair_file
  pair_file="$(write_sparse_pairs)"

  HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
  HIDANN_CONSTRUCTION_FADVISE=0 \
  HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
  run_scoped_time "${CGROUP_LIMIT_BYTES}" "hidann_${DATASET_NAME}_sparse_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      python3 "${ROOT}/construction/run_sigma_pipeline.py" \
        --data "${BASE}" \
        --prefix "${prefix}" \
        --min-P 1 \
        --R "${R}" \
        --L "${L}" \
        --QD "${QD}" \
        --threads "${THREADS}" \
        --build-dir "${BUILD_DIR}" \
        --pq-sampling-rate 0.05 \
        --memory-budget-bytes "${MEMORY_BUDGET_BYTES}" \
        --chunk-budget "${CHUNK_BUDGET_BYTES}" \
        --partition-budget-fraction "${PARTITION_BUDGET_FRACTION}" \
        --cross-pairs-file "${pair_file}" \
    2>&1 | tee "${outer_log}"
}

prepare_layout() {
  require_data
  local graph
  graph="$(detect_full_graph)"
  local construction_dir="${RUN_DIR}/full/construction"
  local layout_dir="${RUN_DIR}/full/layout"
  local layout_prefix="${layout_dir}/${DATASET_PREFIX}_QD${QD}"
  mkdir -p "${layout_dir}"

  cp -f "${construction_dir}/${DATASET_PREFIX}_QD${QD}.pq_compressed.bin" "${layout_prefix}_pq_compressed.bin"
  cp -f "${construction_dir}/${DATASET_PREFIX}_QD${QD}.pq_pivots.bin" "${layout_prefix}_pq_pivots.bin"
  run_cmd "${BUILD_DIR}/tools/create_disk_layout" float "${BASE}" "${graph}" "${layout_prefix}" \
    2>&1 | tee "${layout_dir}/${DATASET_PREFIX}_layout.log"
}

run_search() {
  require_data
  local n
  local effective_threads
  n="$(read_fbin_n "${BASE}")"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "${DATASET_LABEL} HiDANN search")"
  local layout_prefix="${RUN_DIR}/full/layout/${DATASET_PREFIX}_QD${QD}"
  local search_dir="${RUN_DIR}/full/search"
  local cache_file="${search_dir}/cache/${DATASET_PREFIX}_cache.txt"
  local log_file="${search_dir}/${DATASET_PREFIX}_cached_multi_b${BEAM_WIDTH}_t${effective_threads}.log"
  mkdir -p "$(dirname "${cache_file}")"
  seq 0 "$((n - 1))" > "${cache_file}"

  run_scoped_time "${SEARCH_CGROUP_LIMIT_BYTES}" "hidann_${DATASET_NAME}_search_$$" \
    env CACHE_FILE="${cache_file}" \
      BATCH_REFINE_SIZE=32 \
      EARLY_EXIT_STREAK=1 \
      SIGMA_LAYOUT=COMBINED \
      SIGMA_CACHE_LOOKUP_MODE=auto \
      OMP_NUM_THREADS="${effective_threads}" \
      MKL_NUM_THREADS="${effective_threads}" \
      "${BUILD_DIR}/search/ours_search" \
        --data_type float \
        --dist_fn l2 \
        --index_path_prefix "${layout_prefix}" \
        --query_file "${QUERY}" \
        --gt_file "${GT}" \
        --recall_at 10 \
        --search_list "${L_VALUES[@]}" \
        --beamwidth "${BEAM_WIDTH}" \
        --num_threads "${effective_threads}" \
        --num_nodes_to_cache "${n}" \
        --result_path "${search_dir}/${DATASET_PREFIX}_cached_multi" \
        --fail_if_recall_below 0 \
    2>&1 | tee "${log_file}"
}

summarize() {
  run_cmd python3 "${ROOT}/scripts/summarize_dataset_run.py" \
    --run-dir "${RUN_DIR}" \
    --dataset "${DATASET_NAME}" \
    --memory-limit-bytes "${CGROUP_LIMIT_BYTES}"
}

plot() {
  run_cmd python3 "${ROOT}/repro/figures/plot_dataset_reproduction.py" \
    --results-dir "${RUN_DIR}/summary" \
    --out-dir "${RUN_DIR}/plots" \
    --dataset-label "${DATASET_LABEL}" \
    --dataset-slug "${DATASET_NAME}"
}

mark_latest() {
  mkdir -p "${ROOT}/runs"
  ln -sfn "${RUN_DIR}" "${ROOT}/runs/LATEST_${DATASET_NAME}_PIPELINE"
}

case "${MODE}" in
  all)
    require_data
    build_artifact
    run_full_construction
    run_sparse_construction
    prepare_layout
    run_search
    summarize
    plot
    mark_latest
    ;;
  build) build_artifact ;;
  construction) run_full_construction ;;
  sparse) run_sparse_construction ;;
  layout) prepare_layout ;;
  search) run_search ;;
  summarize) summarize ;;
  plots) plot ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac

echo
echo "${DATASET_LABEL} HiDANN pipeline outputs: ${RUN_DIR}"

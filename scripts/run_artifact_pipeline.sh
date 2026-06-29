#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
DATA_DIR="${DATA_DIR:-${ROOT}/data/toy}"
RUN_DIR="${RUN_DIR:-${ROOT}/runs/toy}"

THREADS="${THREADS:-8}"
SEARCH_THREADS="${SEARCH_THREADS:-${THREADS}}"
N="${N:-2048}"
DIM="${DIM:-32}"
QUERIES="${QUERIES:-100}"
P="${P:-4}"
R="${R:-16}"
L="${L:-20}"
QD="${QD:-8}"
K="${K:-10}"
BEAM_WIDTH="${BEAM_WIDTH:-2}"
MIN_RECALL="${MIN_RECALL:-0.50}"
MEMORY_BUDGET_BYTES="${MEMORY_BUDGET_BYTES:-1073741824}"
CHUNK_BUDGET_BYTES="${CHUNK_BUDGET_BYTES:-268435456}"

usage() {
  cat <<EOF
Usage: $0 [all|data|build|construction|sparse-probing|search|clean]

Environment overrides:
  THREADS=${THREADS} N=${N} DIM=${DIM} QUERIES=${QUERIES}
  P=${P} R=${R} L=${L} QD=${QD} K=${K} MIN_RECALL=${MIN_RECALL}
  MEMORY_BUDGET_BYTES=${MEMORY_BUDGET_BYTES}
  CHUNK_BUDGET_BYTES=${CHUNK_BUDGET_BYTES}
  BUILD_DIR=${BUILD_DIR}
  DATA_DIR=${DATA_DIR}
  RUN_DIR=${RUN_DIR}
  CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS:-}"
EOF
}

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

aio_checked_threads() {
  python3 "${ROOT}/scripts/preflight_aio.py" --threads "$1" --label "$2"
}

safe_rm_tree() {
  local path="$1"
  case "${path}" in
    "${ROOT}"/*) rm -rf -- "${path}" ;;
    *) echo "refusing to remove path outside artifact root: ${path}" >&2; return 1 ;;
  esac
}

generate_data() {
  run_cmd python3 "${ROOT}/scripts/make_toy_dataset.py" \
    --out-dir "${DATA_DIR}" \
    --n "${N}" \
    --queries "${QUERIES}" \
    --dim "${DIM}" \
    --gt-k "${K}"
}

build_artifact() {
  # shellcheck disable=SC2206
  local extra_args=(${CMAKE_EXTRA_ARGS:-})
  run_cmd cmake -S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON "${extra_args[@]}"
  run_cmd cmake --build "${BUILD_DIR}" --target our_construction our_pruning reverse_edge_logic create_disk_layout ours_search -j "${THREADS}"
}

write_sparse_pairs() {
  local path="$1"
  mkdir -p "$(dirname "${path}")"
  : > "${path}"
  for ((i = 0; i + 1 < P; ++i)); do
    printf '%d %d\n' "${i}" "$((i + 1))" >> "${path}"
  done
  echo "${path}"
}

run_construction() {
  local label="$1"
  local extra_arg="${2:-}"
  local out_dir="${RUN_DIR}/${label}/construction"
  local prefix="${out_dir}/toy"
  mkdir -p "${out_dir}"

  HIDANN_CONSTRUCTION_STAGE_CACHE_DROP=0 \
  HIDANN_CONSTRUCTION_FADVISE=0 \
  HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=0 \
  run_cmd python3 "${ROOT}/construction/run_sigma_pipeline.py" \
    --data "${DATA_DIR}/base.fbin" \
    --prefix "${prefix}" \
    --P "${P}" \
    --R "${R}" \
    --L "${L}" \
    --QD "${QD}" \
    --threads "${THREADS}" \
    --build-dir "${BUILD_DIR}" \
    --pq-sampling-rate 1.0 \
    --memory-budget-bytes "${MEMORY_BUDGET_BYTES}" \
    --chunk-budget "${CHUNK_BUDGET_BYTES}" \
    ${extra_arg}

  local graph="${prefix}_P${P}_QD${QD}_final.mem.index"
  test -f "${graph}"
  echo "final_graph=${graph}"
}

prepare_layout() {
  local label="$1"
  local prefix="${RUN_DIR}/${label}/construction/toy"
  local graph="${prefix}_P${P}_QD${QD}_final.mem.index"
  local layout_dir="${RUN_DIR}/${label}/layout"
  local layout_prefix="${layout_dir}/toy_QD${QD}"
  mkdir -p "${layout_dir}"

  test -f "${graph}"
  cp -f "${prefix}_QD${QD}.pq_compressed.bin" "${layout_prefix}_pq_compressed.bin"
  cp -f "${prefix}_QD${QD}.pq_pivots.bin" "${layout_prefix}_pq_pivots.bin"
  run_cmd "${BUILD_DIR}/tools/create_disk_layout" float "${DATA_DIR}/base.fbin" "${graph}" "${layout_prefix}" >&2
  echo "${layout_prefix}"
}

run_search() {
  local label="${1:-full}"
  local layout_prefix
  local effective_threads
  layout_prefix="$(prepare_layout "${label}")"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "toy HiDANN search")"
  local search_dir="${RUN_DIR}/${label}/search"
  mkdir -p "${search_dir}"

  SIGMA_LAYOUT=COMBINED \
  BATCH_REFINE_SIZE=16 \
  run_cmd "${BUILD_DIR}/search/ours_search" \
    --data_type float \
    --dist_fn l2 \
    --index_path_prefix "${layout_prefix}" \
    --query_file "${DATA_DIR}/query.fbin" \
    --gt_file "${DATA_DIR}/gt${K}.bin" \
    --recall_at "${K}" \
    --search_list "${L}" \
    --beamwidth "${BEAM_WIDTH}" \
    --num_threads "${effective_threads}" \
    --num_nodes_to_cache 0 \
    --result_path "${search_dir}/toy" \
    --fail_if_recall_below "${MIN_RECALL}"
}

main() {
  local mode="${1:-all}"
  case "${mode}" in
    all)
      generate_data
      build_artifact
      run_construction full
      local sparse_pairs
      sparse_pairs="$(write_sparse_pairs "${RUN_DIR}/sparse/cross_pairs_P${P}.txt")"
      run_construction sparse "--cross-pairs-file=${sparse_pairs}"
      run_search full
      run_search sparse
      ;;
    data) generate_data ;;
    build) build_artifact ;;
    construction) run_construction full ;;
    sparse-probing|sparse-construction)
      sparse_pairs="$(write_sparse_pairs "${RUN_DIR}/sparse/cross_pairs_P${P}.txt")"
      run_construction sparse "--cross-pairs-file=${sparse_pairs}"
      ;;
    search) run_search full ;;
    clean)
      safe_rm_tree "${BUILD_DIR}"
      safe_rm_tree "${RUN_DIR}"
      safe_rm_tree "${DATA_DIR}"
      ;;
    -h|--help|help) usage ;;
    *) usage; exit 2 ;;
  esac
}

main "$@"

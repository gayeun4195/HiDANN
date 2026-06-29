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
RUN_DIR="${2:-${ROOT}/runs/${DATASET_NAME}_comparison_$(date -u +%Y%m%d_%H%M%S)}"
MODE="${3:-all}"

BUILD_DIR="${BUILD_DIR:-${ROOT}/build}"
DISKANN_BUILD_DIR="${DISKANN_BUILD_DIR:-${ROOT}/baselines/DiskANN/build}"
SOGAIC_BUILD_DIR="${SOGAIC_BUILD_DIR:-${ROOT}/baselines/SOGAIC/build}"
PIPEANN_BUILD_DIR="${PIPEANN_BUILD_DIR:-${ROOT}/baselines/PipeANN/build-aio}"

THREADS="${THREADS:-128}"
SEARCH_THREADS="${SEARCH_THREADS:-${THREADS}}"
R="${R:-32}"
L="${L:-50}"
QD="${QD:-512}"
K="${K:-10}"
BEAM_WIDTH="${BEAM_WIDTH:-8}"
MEMORY_BUDGET_BYTES="${MEMORY_BUDGET_BYTES:-323277876}"
CHUNK_BUDGET_BYTES="${CHUNK_BUDGET_BYTES:-161638938}"
CGROUP_LIMIT_BYTES="${CGROUP_LIMIT_BYTES:-860148788}"
SEARCH_CGROUP_LIMIT_BYTES="${SEARCH_CGROUP_LIMIT_BYTES:-1705667336}"
QUALITY_CGROUP_LIMIT_BYTES="${QUALITY_CGROUP_LIMIT_BYTES:-6442450944}"
PIPEANN_CGROUP_LIMIT_BYTES="${PIPEANN_CGROUP_LIMIT_BYTES:-1885671912}"
BUILD_DRAM_GB="${BUILD_DRAM_GB:-0.301076}"
PQ_SAMPLING_RATE="${PQ_SAMPLING_RATE:-0.05}"
DISKANN_PARTITION_SAMPLING_RATE="${DISKANN_PARTITION_SAMPLING_RATE:-${PQ_SAMPLING_RATE}}"
DISKANN_PARTITION_CHUNK_BYTES="${DISKANN_PARTITION_CHUNK_BYTES:-142745899}"
DISKANN_CACHE_NODES="${DISKANN_CACHE_NODES:-15225}"
PIPEANN_SAMPLE_SEED="${PIPEANN_SAMPLE_SEED:-20260602}"
PIPEANN_SAMPLE_CHUNK_MB="${PIPEANN_SAMPLE_CHUNK_MB:-64}"
PIPEANN_BUILD_L="${PIPEANN_BUILD_L:-50}"
PIPEANN_ALPHA="${PIPEANN_ALPHA:-1.2}"
PIPEANN_MEM_L="${PIPEANN_MEM_L:-1}"
SOGAIC_MAX_OVERLAP="${SOGAIC_MAX_OVERLAP:-4.0}"
SOGAIC_EPSILON="${SOGAIC_EPSILON:-1.8}"
SOGAIC_GAMMA="${SOGAIC_GAMMA:-1.0}"
SOGAIC_PARTITION_BUDGET_FRACTION="${SOGAIC_PARTITION_BUDGET_FRACTION:-0.95}"
QUALITY_COMMON_ENTRY="${QUALITY_COMMON_ENTRY:-}"
RUN_HIDANN="${RUN_HIDANN:-1}"
RUN_SPARSE="${RUN_SPARSE:-1}"
RUN_DISKANN="${RUN_DISKANN:-1}"
RUN_SOGAIC="${RUN_SOGAIC:-1}"
RUN_PIPEANN="${RUN_PIPEANN:-1}"
RUN_QUALITY="${RUN_QUALITY:-1}"
RUN_SAME_INDEX_SEARCH="${RUN_SAME_INDEX_SEARCH:-1}"
DISKANN_CLEAN_TEMP="${DISKANN_CLEAN_TEMP:-1}"
DISKANN_KEEP_MEM_INDEX="${DISKANN_KEEP_MEM_INDEX:-${RUN_QUALITY}}"
SOGAIC_CLEAN_TEMP="${SOGAIC_CLEAN_TEMP:-1}"
USE_CGROUP="${USE_CGROUP:-1}"
REUSE="${REUSE:-0}"
L_VALUES=(${L_VALUES:-10 20 30 40 50 75 100 150 200 250 300 350 400 450 500 750 1000})
QUALITY_L_VALUES=(${QUALITY_L_VALUES:-10 20 30 40 50 75 100 150 200 300 400 500})

BASE="${DATA_DIR}/${BASE_FILE}"
QUERY="${DATA_DIR}/${QUERY_FILE}"
GT="${DATA_DIR}/${GT_FILE}"

N=""
D=""

usage() {
  cat <<EOF
Usage: $0 [data_dir] [run_dir] [all|build|hidann|diskann|sogaic|search|pipeann|same_search|quality|summarize|plots]

This runs the artifact comparison for one prepared dataset:
  HiDANN construction/sparse probing/search
  DiskANN common-PQ construction + DiskANN search
  SOGAIC limited-memory construction
  PipeANN search on the DiskANN index
  Same-HiDANN-index DiskANN/PipeANN/HiDANN search comparison
  B=1 single-common-entry memory-index quality evaluation

Generated indexes, logs, CSVs, and plots are written under run_dir.
EOF
}

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

require_executable() {
  if [[ ! -x "$1" ]]; then
    echo "missing executable: $1" >&2
    exit 1
  fi
}

require_data() {
  require_file "${BASE}"
  require_file "${QUERY}"
  require_file "${GT}"
}

read_dataset_meta() {
  read -r N D < <(python3 - "${BASE}" <<'PY'
import struct
import sys
with open(sys.argv[1], "rb") as handle:
    n, d = struct.unpack("<II", handle.read(8))
print(n, d)
PY
)
}

now_seconds() {
  python3 - <<'PY'
import time
print(f"{time.time():.6f}")
PY
}

elapsed_seconds() {
  python3 - "$1" "$2" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
}

sum_seconds() {
  awk 'BEGIN { total = 0; for (i = 1; i < ARGC; i++) total += ARGV[i]; printf "%.6f", total; exit }' "$@"
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

detect_pybind11_cmake_dir() {
  if [[ -n "${pybind11_DIR:-}" ]]; then
    echo "${pybind11_DIR}"
    return 0
  fi
  if [[ -n "${PYBIND11_DIR:-}" ]]; then
    echo "${PYBIND11_DIR}"
    return 0
  fi
  python3 -m pybind11 --cmakedir 2>/dev/null
}

run_logged() {
  local log_file="$1"
  local memory_limit="$2"
  local unit_name="$3"
  shift 3
  mkdir -p "$(dirname "${log_file}")"
  {
    echo "============================================================"
    echo "start_time=$(date -Is)"
    printf 'command='
    printf '%q ' "$@"
    echo
    echo "memory_limit_bytes=${memory_limit}"
  } | tee -a "${log_file}"
  run_scoped_time "${memory_limit}" "${unit_name}" "$@" 2>&1 | tee -a "${log_file}"
}

build_hidann() {
  run_cmd cmake -S "${ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
  run_cmd cmake --build "${BUILD_DIR}" --target \
    our_construction our_pruning reverse_edge_logic create_disk_layout \
    evaluate_memory_index_quality ours_search -j "${THREADS}"
}

build_baselines() {
  run_cmd cmake -S "${ROOT}/baselines/DiskANN" -B "${DISKANN_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
  run_cmd cmake --build "${DISKANN_BUILD_DIR}" --target \
    build_memory_index search_disk_index partition_with_ram_budget \
    extract_shard_data_from_ids merge_shards create_disk_layout \
    generate_pq gen_random_slice -j "${THREADS}"

  run_cmd cmake -S "${ROOT}/baselines/SOGAIC" -B "${SOGAIC_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON
  run_cmd cmake --build "${SOGAIC_BUILD_DIR}" --target \
    build_memory_index sogaic_partition_data merge_shards -j "${THREADS}"

  local pipeann_extra_args=()
  local pybind11_cmake_dir
  if pybind11_cmake_dir="$(detect_pybind11_cmake_dir)" && [[ -n "${pybind11_cmake_dir}" ]]; then
    pipeann_extra_args+=("-Dpybind11_DIR=${pybind11_cmake_dir}")
  fi
  if [[ -n "${EIGEN3_INCLUDE_DIR:-}" ]]; then
    pipeann_extra_args+=("-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR}")
  elif [[ -n "${EIGEN3_INCLUDE_DIRS:-}" ]]; then
    pipeann_extra_args+=("-DEIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIRS%%[:;]*}")
  fi
  # shellcheck disable=SC2206
  local user_pipeann_args=(${PIPEANN_CMAKE_EXTRA_ARGS:-})
  pipeann_extra_args+=("${user_pipeann_args[@]}")

  run_cmd cmake -S "${ROOT}/baselines/PipeANN" -B "${PIPEANN_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release -DUSE_AIO=ON "${pipeann_extra_args[@]}"
  run_cmd cmake --build "${PIPEANN_BUILD_DIR}" --target \
    search_disk_index build_memory_index gen_random_slice -j "${THREADS}"
}

run_hidann_pipeline() {
  if [[ "${RUN_HIDANN}" != "1" ]]; then
    echo "RUN_HIDANN=${RUN_HIDANN}; skipping HiDANN"
    return
  fi
  if [[ "${REUSE}" == "1" && -f "${RUN_DIR}/hidann/summary/pipeline_status.json" ]]; then
    echo "reusing HiDANN outputs in ${RUN_DIR}/hidann"
    return
  fi
  USE_CGROUP="${USE_CGROUP}" \
  RUN_SPARSE="${RUN_SPARSE}" \
  REUSE="${REUSE}" \
  DATASET_NAME="${DATASET_NAME}" \
  DATASET_PREFIX="${DATASET_PREFIX}" \
  DATASET_LABEL="${DATASET_LABEL}" \
  BASE_FILE="${BASE_FILE}" \
  QUERY_FILE="${QUERY_FILE}" \
  GT_FILE="${GT_FILE}" \
  BUILD_DIR="${BUILD_DIR}" \
  THREADS="${THREADS}" \
  SEARCH_THREADS="${SEARCH_THREADS}" \
  R="${R}" \
  L="${L}" \
  QD="${QD}" \
  BEAM_WIDTH="${BEAM_WIDTH}" \
  MEMORY_BUDGET_BYTES="${MEMORY_BUDGET_BYTES}" \
  CHUNK_BUDGET_BYTES="${CHUNK_BUDGET_BYTES}" \
  CGROUP_LIMIT_BYTES="${CGROUP_LIMIT_BYTES}" \
  SEARCH_CGROUP_LIMIT_BYTES="${SEARCH_CGROUP_LIMIT_BYTES}" \
    "${ROOT}/scripts/run_dataset_pipeline.sh" "${DATA_DIR}" "${RUN_DIR}/hidann" all
}

diskann_prefix() {
  echo "${RUN_DIR}/baselines/diskann/construction/${DATASET_PREFIX}"
}

hidann_layout_prefix() {
  echo "${RUN_DIR}/hidann/full/layout/${DATASET_PREFIX}_QD${QD}"
}

run_diskann_common_pq_construction() {
  if [[ "${RUN_DISKANN}" != "1" ]]; then
    echo "RUN_DISKANN=${RUN_DISKANN}; skipping DiskANN construction"
    return
  fi
  require_data
  read_dataset_meta
  local prefix temp_prefix mem_index disk_index medoids centroids sample_prefix log_file
  local sub_r partition_start partition_end partition_time pq_start pq_end pq_time
  local shard_start shard_end shard_time merge_start merge_end merge_time
  local layout_start layout_end layout_time sample_start sample_end sample_time
  local pipeline_start pipeline_end pipeline_time merged_vamana_time excluding_layout_time
  prefix="$(diskann_prefix)"
  temp_prefix="${prefix}_mem.index_tempFiles"
  mem_index="${prefix}_mem.index"
  disk_index="${prefix}_disk.index"
  medoids="${disk_index}_medoids.bin"
  centroids="${disk_index}_centroids.bin"
  sample_prefix="${prefix}_sample"
  log_file="${RUN_DIR}/baselines/diskann/construction/${DATASET_PREFIX}_diskann_common_pq.log"
  mkdir -p "$(dirname "${prefix}")"

  if [[ "${REUSE}" == "1" && -s "${disk_index}" && -s "${prefix}_pq_compressed.bin" && -s "${prefix}_pq_pivots.bin" ]]; then
    echo "reusing DiskANN common-PQ construction in $(dirname "${prefix}")"
    return
  fi

  require_executable "${DISKANN_BUILD_DIR}/apps/utils/generate_pq"
  require_executable "${DISKANN_BUILD_DIR}/apps/utils/partition_with_ram_budget"
  require_executable "${DISKANN_BUILD_DIR}/apps/build_memory_index"
  require_executable "${DISKANN_BUILD_DIR}/apps/utils/merge_shards"
  require_executable "${DISKANN_BUILD_DIR}/apps/utils/create_disk_layout"
  require_executable "${DISKANN_BUILD_DIR}/apps/utils/gen_random_slice"
  require_executable "${ROOT}/scripts/extract_shard_data_from_ids.py"

  sub_r=$((2 * R / 3))
  if [[ "${sub_r}" -lt 16 ]]; then sub_r=16; fi
  if [[ "${sub_r}" -gt "${R}" ]]; then sub_r="${R}"; fi

  : > "${log_file}"
  {
    echo "[diskann-common-pq] data=${BASE}"
    echo "[diskann-common-pq] prefix=${prefix}"
    echo "[diskann-common-pq] N=${N} D=${D} R=${R} L=${L} sub_R=${sub_r} QD=${QD}"
    echo "[diskann-common-pq] build_budget_gb=${BUILD_DRAM_GB}"
    echo "[diskann-common-pq] pq_sampling_rate=${PQ_SAMPLING_RATE}"
    echo "[diskann-common-pq] partition_sampling_rate=${DISKANN_PARTITION_SAMPLING_RATE}"
    echo "[diskann-common-pq] partition_chunk_bytes=${DISKANN_PARTITION_CHUNK_BYTES}"
  } | tee -a "${log_file}"

  rm -f "${temp_prefix}"_subshard-* "${temp_prefix}_centroids.bin" \
    "${mem_index}" "${mem_index}.data" "${mem_index}.tags" "${disk_index}" \
    "${disk_index}_medoids.bin" "${disk_index}_centroids.bin"

  pipeline_start="$(now_seconds)"

  pq_start="$(now_seconds)"
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_pq_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      DISKANN_CHUNK_BYTES="${CHUNK_BUDGET_BYTES}" \
      "${DISKANN_BUILD_DIR}/apps/utils/generate_pq" float "${BASE}" "${prefix}" "${QD}" "${PQ_SAMPLING_RATE}" 0
  pq_end="$(now_seconds)"
  pq_time="$(elapsed_seconds "${pq_start}" "${pq_end}")"

  partition_start="$(now_seconds)"
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_part_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
      DISKANN_CHUNK_BYTES="${DISKANN_PARTITION_CHUNK_BYTES}" \
      DISKANN_PARTITION_RAM_BUDGET_FRACTION=0.95 \
      DISKANN_PARTITION_CHUNK_BYTES="${DISKANN_PARTITION_CHUNK_BYTES}" \
      DISKANN_PARTITION_SAMPLING_RATE="${DISKANN_PARTITION_SAMPLING_RATE}" \
      "${DISKANN_BUILD_DIR}/apps/utils/partition_with_ram_budget" \
        float "${BASE}" "${temp_prefix}" "${DISKANN_PARTITION_SAMPLING_RATE}" "${BUILD_DRAM_GB}" "${sub_r}" 2
  partition_end="$(now_seconds)"
  partition_time="$(elapsed_seconds "${partition_start}" "${partition_end}")"

  shopt -s nullglob
  local idmaps=("${temp_prefix}"_subshard-*_ids_uint32.bin)
  shopt -u nullglob
  if [[ "${#idmaps[@]}" -eq 0 ]]; then
    echo "No DiskANN shard id maps were produced" >&2
    exit 1
  fi
  local num_parts="${#idmaps[@]}"
  echo "[diskann-common-pq] num_parts=${num_parts}" | tee -a "${log_file}"
  if [[ -f "${temp_prefix}_centroids.bin" ]]; then
    mv -f "${temp_prefix}_centroids.bin" "${centroids}"
  fi

  shard_start="$(now_seconds)"
  for ((p = 0; p < num_parts; p++)); do
    local shard_base="${temp_prefix}_subshard-${p}.bin"
    local shard_ids="${temp_prefix}_subshard-${p}_ids_uint32.bin"
    local shard_index="${temp_prefix}_subshard-${p}_mem.index"
    echo "[diskann-common-pq] shard ${p}/${num_parts}" | tee -a "${log_file}"
    run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_extract_${p}_$$" \
      "${ROOT}/scripts/extract_shard_data_from_ids.py" "${BASE}" "${shard_ids}" "${shard_base}"
    run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_build_${p}_$$" \
      env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OMP_PROC_BIND=close OMP_PLACES=cores \
        "${DISKANN_BUILD_DIR}/apps/build_memory_index" \
          --data_type float \
          --dist_fn l2 \
          --index_path_prefix "${shard_index}" \
          --data_path "${shard_base}" \
          -R "${sub_r}" \
          -L "${L}" \
          --alpha 1.2 \
          -T "${THREADS}"
    rm -f "${shard_base}"
  done
  shard_end="$(now_seconds)"
  shard_time="$(elapsed_seconds "${shard_start}" "${shard_end}")"

  merge_start="$(now_seconds)"
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_merge_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
      "${DISKANN_BUILD_DIR}/apps/utils/merge_shards" \
        "${temp_prefix}_subshard-" \
        "_mem.index" \
        "${temp_prefix}_subshard-" \
        "_ids_uint32.bin" \
        "${num_parts}" \
        "${R}" \
        "${mem_index}" \
        "${medoids}"
  merge_end="$(now_seconds)"
  merge_time="$(elapsed_seconds "${merge_start}" "${merge_end}")"

  merged_vamana_time="$(sum_seconds "${partition_time}" "${shard_time}" "${merge_time}")"
  excluding_layout_time="$(sum_seconds "${pq_time}" "${merged_vamana_time}")"

  layout_start="$(now_seconds)"
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_layout_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
      "${DISKANN_BUILD_DIR}/apps/utils/create_disk_layout" float "${BASE}" "${mem_index}" "${disk_index}"
  layout_end="$(now_seconds)"
  layout_time="$(elapsed_seconds "${layout_start}" "${layout_end}")"

  local sample_rate
  sample_rate="$(python3 - "${N}" <<'PY'
import math
import sys
n = int(sys.argv[1])
print("{:.12g}".format(min(math.ceil(n * 0.1), 100000) / n))
PY
)"
  sample_start="$(now_seconds)"
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_sample_$$" \
    env OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" \
      "${DISKANN_BUILD_DIR}/apps/utils/gen_random_slice" float "${BASE}" "${sample_prefix}" "${sample_rate}"
  sample_end="$(now_seconds)"
  sample_time="$(elapsed_seconds "${sample_start}" "${sample_end}")"

  local clean_time="0.000000"
  if [[ "${DISKANN_CLEAN_TEMP}" == "1" ]]; then
    local clean_start clean_end
    clean_start="$(now_seconds)"
    if [[ ! -s "${disk_index}" || ! -s "${medoids}" ]]; then
      echo "refusing to clean DiskANN temp files because final outputs are missing" >&2
      exit 1
    fi
    if [[ "${DISKANN_KEEP_MEM_INDEX}" != "1" ]]; then
      rm -f "${mem_index}" "${mem_index}.data" "${mem_index}.tags"
    fi
    for ((p = 0; p < num_parts; p++)); do
      rm -f \
        "${temp_prefix}_subshard-${p}.bin" \
        "${temp_prefix}_subshard-${p}_ids_uint32.bin" \
        "${temp_prefix}_subshard-${p}_mem.index" \
        "${temp_prefix}_subshard-${p}_mem.index.data" \
        "${temp_prefix}_subshard-${p}_mem.index.tags"
    done
    clean_end="$(now_seconds)"
    clean_time="$(elapsed_seconds "${clean_start}" "${clean_end}")"
  fi

  pipeline_end="$(now_seconds)"
  pipeline_time="$(elapsed_seconds "${pipeline_start}" "${pipeline_end}")"
  {
    echo "Time for generating quantized data: ${pq_time} seconds"
    echo "Time for partitioning data : ${partition_time} seconds"
    echo "Time for building indices on shards: ${shard_time} seconds"
    echo "Time for merging indices: ${merge_time} seconds"
    echo "Time for building merged vamana index: ${merged_vamana_time} seconds"
    echo "Time for DiskANN construction excluding disk layout: ${excluding_layout_time} seconds"
    echo "Time for generating disk layout: ${layout_time} seconds"
    echo "Time for generating sample data: ${sample_time} seconds"
    echo "Time for cleaning temporary files: ${clean_time} seconds"
    echo "Indexing time: ${pipeline_time}"
    echo "[diskann-common-pq] measured_construction_excluding_layout_seconds=${excluding_layout_time}"
    echo "[diskann-common-pq] measured_merged_vamana_seconds=${merged_vamana_time}"
    echo "[diskann-common-pq] done"
  } | tee -a "${log_file}"
}

run_sogaic_construction() {
  if [[ "${RUN_SOGAIC}" != "1" ]]; then
    echo "RUN_SOGAIC=${RUN_SOGAIC}; skipping SOGAIC"
    return
  fi
  require_data
  local out_dir prefix log_file
  out_dir="${RUN_DIR}/baselines/sogaic/construction"
  prefix="${out_dir}/${DATASET_PREFIX}"
  log_file="${out_dir}/${DATASET_PREFIX}_sogaic_outer.log"
  mkdir -p "${out_dir}"
  if [[ "${REUSE}" == "1" && -s "${prefix}_merged.index" ]]; then
    echo "reusing SOGAIC construction in ${out_dir}"
    return
  fi
  require_executable "${ROOT}/baselines/SOGAIC/scripts/run_sogaic_pipeline_limited.sh"
  SOGAIC_CLEAN_TEMP="${SOGAIC_CLEAN_TEMP}" \
  run_logged "${log_file}" "${CGROUP_LIMIT_BYTES}" "hidann_artifact_sogaic_$$" \
    "${ROOT}/baselines/SOGAIC/scripts/run_sogaic_pipeline_limited.sh" \
      "${BASE}" "${prefix}" \
      --datatype float \
      --budget-ratio 0.10 \
      --overlap "${SOGAIC_MAX_OVERLAP}" \
      --epsilon "${SOGAIC_EPSILON}" \
      --gamma "${SOGAIC_GAMMA}" \
      --partition-budget-fraction "${SOGAIC_PARTITION_BUDGET_FRACTION}" \
      --memory-budget-bytes "${MEMORY_BUDGET_BYTES}" \
      --sampling "${PQ_SAMPLING_RATE}" \
      --threads "${THREADS}" \
      --R "${R}" \
      --L "${L}" \
      --chunk-budget "${CHUNK_BUDGET_BYTES}" \
      --clean-temp "${SOGAIC_CLEAN_TEMP}"
}

run_diskann_search() {
  if [[ "${RUN_DISKANN}" != "1" ]]; then
    echo "RUN_DISKANN=${RUN_DISKANN}; skipping DiskANN search"
    return
  fi
  local prefix search_dir log_file effective_threads
  prefix="$(diskann_prefix)"
  search_dir="${RUN_DIR}/baselines/diskann/search"
  log_file="${search_dir}/${DATASET_PREFIX}_diskann_search.log"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "${DATASET_LABEL} DiskANN search")"
  mkdir -p "${search_dir}"
  require_file "${prefix}_disk.index"
  require_file "${prefix}_pq_compressed.bin"
  require_file "${prefix}_pq_pivots.bin"
  require_executable "${DISKANN_BUILD_DIR}/apps/search_disk_index"
  if [[ "${REUSE}" == "1" && -f "${log_file}" ]]; then
    echo "reusing DiskANN search log ${log_file}"
    return
  fi
  run_logged "${log_file}" "${SEARCH_CGROUP_LIMIT_BYTES}" "hidann_artifact_diskann_search_$$" \
    "${DISKANN_BUILD_DIR}/apps/search_disk_index" \
      --data_type float \
      --dist_fn l2 \
      --index_path_prefix "${prefix}" \
      --query_file "${QUERY}" \
      --gt_file "${GT}" \
      --recall_at "${K}" \
      --search_list "${L_VALUES[@]}" \
      --beamwidth "${BEAM_WIDTH}" \
      --num_threads "${effective_threads}" \
      --num_nodes_to_cache "${DISKANN_CACHE_NODES}" \
      --result_path "${search_dir}/${DATASET_PREFIX}_diskann" \
      --fail_if_recall_below 0
}

pipeann_prefix() {
  echo "${RUN_DIR}/baselines/pipeann/work/${DATASET_PREFIX}_idx"
}

pipeann_hidann_prefix() {
  echo "${RUN_DIR}/same_hidann_index/pipeann_work/${DATASET_PREFIX}_idx"
}

generate_pipeann_sample() {
  local prefix="$1"
  local target_nodes="$2"
  python3 - "${BASE}" "${prefix}_SAMPLE" "${target_nodes}" "${PIPEANN_SAMPLE_SEED}" "${PIPEANN_SAMPLE_CHUNK_MB}" "${DATASET_NAME}" <<'PY'
import hashlib
import random
import struct
import sys
from pathlib import Path

base_path = Path(sys.argv[1])
out_prefix = Path(sys.argv[2])
target_nodes = int(sys.argv[3])
seed_text = sys.argv[4]
chunk_mb = int(sys.argv[5])
dataset_name = sys.argv[6]

with base_path.open("rb") as handle:
    n, d = struct.unpack("<II", handle.read(8))
target_nodes = max(0, min(target_nodes, n))
if target_nodes <= 0:
    raise SystemExit("PipeANN sample target is zero")

seed_material = f"{seed_text}\0{dataset_name}\0{n}\0{d}".encode()
derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little")
rng = random.Random(derived_seed)
selected = None if target_nodes == n else set(rng.sample(range(n), target_nodes))
row_size = d * 4
chunk_rows = max(1, (chunk_mb * 1024 * 1024) // row_size)
data_path = Path(str(out_prefix) + "_data.bin")
ids_path = Path(str(out_prefix) + "_ids.bin")
data_path.parent.mkdir(parents=True, exist_ok=True)

written = 0
with base_path.open("rb") as src, data_path.open("wb") as data_out, ids_path.open("wb") as ids_out:
    src.seek(8)
    data_out.write(struct.pack("<II", target_nodes, d))
    ids_out.write(struct.pack("<II", target_nodes, 1))
    base = 0
    while base < n:
        count = min(chunk_rows, n - base)
        buf = src.read(count * row_size)
        if len(buf) != count * row_size:
            raise SystemExit(f"short read at row {base}: {base_path}")
        view = memoryview(buf)
        for local in range(count):
            idx = base + local
            if selected is None or idx in selected:
                off = local * row_size
                data_out.write(view[off : off + row_size])
                ids_out.write(struct.pack("<I", idx))
                written += 1
        base += count
if written != target_nodes:
    raise SystemExit(f"sample wrote {written} rows, expected {target_nodes}")
print(f"pipeann_deterministic_sample_derived_seed={derived_seed}")
print(f"pipeann_deterministic_sample_rows={written}")
print(f"pipeann_deterministic_sample_dim={d}")
print(f"pipeann_deterministic_sample_data={data_path}")
print(f"pipeann_deterministic_sample_ids={ids_path}")
PY
}

prepare_pipeann_workdir_from_prefix() {
  local src="$1"
  local target="$2"
  local src_abs
  mkdir -p "$(dirname "${target}")"
  require_file "${src}_disk.index"
  require_file "${src}_pq_pivots.bin"
  require_file "${src}_pq_compressed.bin"
  src_abs="$(realpath "${src}_disk.index")"
  ln -sfn "${src_abs}" "${target}_disk.index"
  src_abs="$(realpath "${src}_pq_pivots.bin")"
  ln -sfn "${src_abs}" "${target}_pq_pivots.bin"
  src_abs="$(realpath "${src}_pq_compressed.bin")"
  ln -sfn "${src_abs}" "${target}_pq_compressed.bin"
  if [[ -f "${src}_disk.index_medoids.bin" ]]; then
    src_abs="$(realpath "${src}_disk.index_medoids.bin")"
    ln -sfn "${src_abs}" "${target}_disk.index_medoids.bin"
  fi
  if [[ -f "${src}_medoids.bin" ]]; then
    src_abs="$(realpath "${src}_medoids.bin")"
    ln -sfn "${src_abs}" "${target}_medoids.bin"
  fi
}

prepare_pipeann_workdir() {
  prepare_pipeann_workdir_from_prefix "$(diskann_prefix)" "$(pipeann_prefix)"
}

run_pipeann_search() {
  if [[ "${RUN_PIPEANN}" != "1" ]]; then
    echo "RUN_PIPEANN=${RUN_PIPEANN}; skipping PipeANN"
    return
  fi
  require_data
  read_dataset_meta
  local prefix search_dir build_log log_file sample_target effective_threads
  prefix="$(pipeann_prefix)"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "${DATASET_LABEL} PipeANN search")"
  search_dir="${RUN_DIR}/baselines/pipeann/search"
  build_log="${search_dir}/${DATASET_PREFIX}_pipeann_build.log"
  log_file="${search_dir}/${DATASET_PREFIX}_pipeann_search.log"
  mkdir -p "${search_dir}"
  prepare_pipeann_workdir
  require_executable "${PIPEANN_BUILD_DIR}/tests/build_memory_index"
  require_executable "${PIPEANN_BUILD_DIR}/tests/search_disk_index"
  sample_target="${DISKANN_CACHE_NODES}"
  if [[ "${REUSE}" == "1" && -s "${prefix}_mem.index" && -f "${log_file}" ]]; then
    echo "reusing PipeANN search in ${search_dir}"
    return
  fi
  : > "${build_log}"
  {
    echo "pipeann_sample_mode=deterministic"
    echo "pipeann_sample_seed=${PIPEANN_SAMPLE_SEED}"
    echo "pipeann_sample_target_nodes=${sample_target}"
    echo "pipeann_mem_L=${PIPEANN_MEM_L}"
  } | tee -a "${build_log}"
  if [[ ! -s "${prefix}_mem.index" ]]; then
    generate_pipeann_sample "${prefix}" "${sample_target}" 2>&1 | tee -a "${build_log}"
    run_logged "${build_log}" "${PIPEANN_CGROUP_LIMIT_BYTES}" "hidann_artifact_pipeann_build_$$" \
      "${PIPEANN_BUILD_DIR}/tests/build_memory_index" \
        float "${prefix}_SAMPLE_data.bin" "${prefix}_SAMPLE_ids.bin" \
        "${prefix}_mem.index" "${R}" "${PIPEANN_BUILD_L}" "${PIPEANN_ALPHA}" "${THREADS}" l2
  fi
  : > "${log_file}"
  cat "${build_log}" >> "${log_file}"
  run_logged "${log_file}" "${PIPEANN_CGROUP_LIMIT_BYTES}" "hidann_artifact_pipeann_search_$$" \
    "${PIPEANN_BUILD_DIR}/tests/search_disk_index" \
      float "${prefix}" "${effective_threads}" "${BEAM_WIDTH}" \
      "${QUERY}" "${GT}" "${K}" l2 pq 2 "${PIPEANN_MEM_L}" "${L_VALUES[@]}"
}

run_hidann_index_diskann_search() {
  if [[ "${RUN_SAME_INDEX_SEARCH}" != "1" ]]; then
    echo "RUN_SAME_INDEX_SEARCH=${RUN_SAME_INDEX_SEARCH}; skipping HiDANN-index DiskANN search"
    return
  fi
  local prefix search_dir log_file effective_threads
  prefix="$(hidann_layout_prefix)"
  search_dir="${RUN_DIR}/same_hidann_index/diskann_search"
  log_file="${search_dir}/${DATASET_PREFIX}_hidann_index_diskann_search.log"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "${DATASET_LABEL} same-index DiskANN search")"
  mkdir -p "${search_dir}"
  require_file "${prefix}_disk.index"
  require_file "${prefix}_pq_compressed.bin"
  require_file "${prefix}_pq_pivots.bin"
  require_executable "${DISKANN_BUILD_DIR}/apps/search_disk_index"
  if [[ "${REUSE}" == "1" && -f "${log_file}" ]]; then
    echo "reusing HiDANN-index DiskANN search log ${log_file}"
    return
  fi
  : > "${log_file}"
  run_logged "${log_file}" "${SEARCH_CGROUP_LIMIT_BYTES}" "hidann_artifact_sameidx_diskann_$$" \
    "${DISKANN_BUILD_DIR}/apps/search_disk_index" \
      --data_type float \
      --dist_fn l2 \
      --index_path_prefix "${prefix}" \
      --query_file "${QUERY}" \
      --gt_file "${GT}" \
      --recall_at "${K}" \
      --search_list "${L_VALUES[@]}" \
      --beamwidth "${BEAM_WIDTH}" \
      --num_threads "${effective_threads}" \
      --num_nodes_to_cache "${DISKANN_CACHE_NODES}" \
      --result_path "${search_dir}/${DATASET_PREFIX}_hidann_index_diskann" \
      --fail_if_recall_below 0
}

run_hidann_index_pipeann_search() {
  if [[ "${RUN_SAME_INDEX_SEARCH}" != "1" ]]; then
    echo "RUN_SAME_INDEX_SEARCH=${RUN_SAME_INDEX_SEARCH}; skipping HiDANN-index PipeANN search"
    return
  fi
  require_data
  read_dataset_meta
  local src_prefix prefix search_dir build_log log_file sample_target effective_threads
  src_prefix="$(hidann_layout_prefix)"
  prefix="$(pipeann_hidann_prefix)"
  effective_threads="$(aio_checked_threads "${SEARCH_THREADS}" "${DATASET_LABEL} same-index PipeANN search")"
  search_dir="${RUN_DIR}/same_hidann_index/pipeann_search"
  build_log="${search_dir}/${DATASET_PREFIX}_hidann_index_pipeann_build.log"
  log_file="${search_dir}/${DATASET_PREFIX}_hidann_index_pipeann_search.log"
  mkdir -p "${search_dir}"
  prepare_pipeann_workdir_from_prefix "${src_prefix}" "${prefix}"
  require_executable "${PIPEANN_BUILD_DIR}/tests/build_memory_index"
  require_executable "${PIPEANN_BUILD_DIR}/tests/search_disk_index"
  sample_target="${DISKANN_CACHE_NODES}"
  if [[ "${REUSE}" == "1" && -s "${prefix}_mem.index" && -f "${log_file}" ]]; then
    echo "reusing HiDANN-index PipeANN search in ${search_dir}"
    return
  fi
  : > "${build_log}"
  {
    echo "pipeann_sample_mode=deterministic"
    echo "pipeann_sample_seed=${PIPEANN_SAMPLE_SEED}"
    echo "pipeann_sample_target_nodes=${sample_target}"
    echo "pipeann_mem_L=${PIPEANN_MEM_L}"
    echo "source_index=HiDANN"
  } | tee -a "${build_log}"
  if [[ ! -s "${prefix}_mem.index" ]]; then
    generate_pipeann_sample "${prefix}" "${sample_target}" 2>&1 | tee -a "${build_log}"
    run_logged "${build_log}" "${PIPEANN_CGROUP_LIMIT_BYTES}" "hidann_artifact_sameidx_pipeann_build_$$" \
      "${PIPEANN_BUILD_DIR}/tests/build_memory_index" \
        float "${prefix}_SAMPLE_data.bin" "${prefix}_SAMPLE_ids.bin" \
        "${prefix}_mem.index" "${R}" "${PIPEANN_BUILD_L}" "${PIPEANN_ALPHA}" "${THREADS}" l2
  fi
  : > "${log_file}"
  cat "${build_log}" >> "${log_file}"
  run_logged "${log_file}" "${PIPEANN_CGROUP_LIMIT_BYTES}" "hidann_artifact_sameidx_pipeann_$$" \
    "${PIPEANN_BUILD_DIR}/tests/search_disk_index" \
      float "${prefix}" "${effective_threads}" "${BEAM_WIDTH}" \
      "${QUERY}" "${GT}" "${K}" l2 pq 2 "${PIPEANN_MEM_L}" "${L_VALUES[@]}"
}

run_same_index_search() {
  run_hidann_index_diskann_search
  run_hidann_index_pipeann_search
}

find_hidann_memory_index() {
  find "${RUN_DIR}/hidann/full/construction" -maxdepth 1 -type f \
    -name "${DATASET_PREFIX}_P*_QD${QD}_final.mem.index" 2>/dev/null | sort | tail -n 1
}

quality_common_entry() {
  if [[ -n "${QUALITY_COMMON_ENTRY}" ]]; then
    echo "${QUALITY_COMMON_ENTRY}"
    return
  fi
  local medoid_file="${RUN_DIR}/quality/${DATASET_PREFIX}_common_medoid.txt"
  if [[ ! -s "${medoid_file}" ]]; then
    run_cmd python3 "${ROOT}/scripts/compute_fbin_medoid.py" \
      --base-file "${BASE}" \
      --out-file "${medoid_file}" >/dev/null
  fi
  tr -d '[:space:]' < "${medoid_file}"
}

run_memory_quality() {
  local method="$1"
  local method_id="$2"
  local index_path="$3"
  local out_csv="$4"
  local log_file="$5"
  local start_node="$6"

  : > "${log_file}"
  run_logged "${log_file}" "${QUALITY_CGROUP_LIMIT_BYTES}" "hidann_artifact_quality_${method_id}_$$" \
    "${BUILD_DIR}/tools/evaluate_memory_index_quality" \
      --data_type float \
      --dist_fn l2 \
      --base_file "${BASE}" \
      --index_path_prefix "${index_path}" \
      --query_file "${QUERY}" \
      --gt_file "${GT}" \
      --recall_at "${K}" \
      --search_list "${QUALITY_L_VALUES[@]}" \
      --num_threads "${THREADS}" \
      --dataset "${DATASET_NAME}" \
      --method "${method}" \
      --method_id "${method_id}" \
      --entry_protocol single_common_entry \
      --start_node "${start_node}" \
      --metric_source memory_index_hop \
      --out_csv "${out_csv}"
}

run_quality() {
  if [[ "${RUN_QUALITY}" != "1" ]]; then
    echo "RUN_QUALITY=${RUN_QUALITY}; skipping B=1 construction-quality evaluation"
    return
  fi
  require_data
  local quality_dir disk_prefix disk_mem sogaic_index hidann_index common_entry
  quality_dir="${RUN_DIR}/quality"
  disk_prefix="$(diskann_prefix)"
  disk_mem="${disk_prefix}_mem.index"
  sogaic_index="${RUN_DIR}/baselines/sogaic/construction/${DATASET_PREFIX}_merged.index"
  hidann_index="$(find_hidann_memory_index)"
  mkdir -p "${quality_dir}"

  if [[ "${REUSE}" == "1" && -s "${RUN_DIR}/summary/index_quality.csv" ]]; then
    echo "reusing ${DATASET_LABEL} quality summary ${RUN_DIR}/summary/index_quality.csv"
    return
  fi

  require_executable "${BUILD_DIR}/tools/evaluate_memory_index_quality"
  common_entry="$(quality_common_entry)"
  echo "[quality] common_entry_source=centroid_nearest_base_vector"
  echo "[quality] common_entry_start_node=${common_entry}"

  if [[ ! -s "${disk_mem}" ]]; then
    echo "[quality] DiskANN memory graph is required for single-common-entry quality evaluation."
    echo "[quality] Rebuilding DiskANN common-PQ construction while keeping the merged memory graph."
    local old_clean old_keep old_reuse old_run_diskann
    old_clean="${DISKANN_CLEAN_TEMP}"
    old_keep="${DISKANN_KEEP_MEM_INDEX}"
    old_reuse="${REUSE}"
    old_run_diskann="${RUN_DISKANN}"
    DISKANN_CLEAN_TEMP=1
    DISKANN_KEEP_MEM_INDEX=1
    REUSE=0
    RUN_DISKANN=1
    run_diskann_common_pq_construction
    DISKANN_CLEAN_TEMP="${old_clean}"
    DISKANN_KEEP_MEM_INDEX="${old_keep}"
    REUSE="${old_reuse}"
    RUN_DISKANN="${old_run_diskann}"
  fi

  if [[ -s "${disk_mem}" ]]; then
    run_memory_quality \
      "DiskANN" \
      "DiskANN" \
      "${disk_mem}" \
      "${quality_dir}/diskann_memory_quality.csv" \
      "${quality_dir}/diskann_memory_quality.log" \
      "${common_entry}"
  else
    echo "missing DiskANN memory index for common-entry quality evaluation: ${disk_mem}" >&2
  fi

  if [[ -s "${sogaic_index}" ]]; then
    run_memory_quality \
      "SOGAIC-style" \
      "SOGAIC" \
      "${sogaic_index}" \
      "${quality_dir}/sogaic_memory_quality.csv" \
      "${quality_dir}/sogaic_memory_quality.log" \
      "${common_entry}"
  else
    echo "missing SOGAIC index for quality evaluation: ${sogaic_index}" >&2
  fi

  if [[ -n "${hidann_index}" && -s "${hidann_index}" ]]; then
    run_memory_quality \
      "HiDANN" \
      "HiDANN" \
      "${hidann_index}" \
      "${quality_dir}/hidann_memory_quality.csv" \
      "${quality_dir}/hidann_memory_quality.log" \
      "${common_entry}"
  else
    echo "missing HiDANN memory index for quality evaluation under ${RUN_DIR}/hidann/full/construction" >&2
  fi

  run_cmd python3 "${ROOT}/scripts/summarize_dataset_quality.py" \
    --run-dir "${RUN_DIR}" \
    --quality-dir "${quality_dir}" \
    --out-csv "${RUN_DIR}/summary/index_quality.csv"
}

summarize() {
  run_cmd python3 "${ROOT}/scripts/summarize_dataset_comparison.py" \
    --run-dir "${RUN_DIR}" \
    --dataset "${DATASET_NAME}" \
    --dataset-prefix "${DATASET_PREFIX}" \
    --memory-limit-bytes "${CGROUP_LIMIT_BYTES}"
  if [[ -d "${RUN_DIR}/quality" ]]; then
    run_cmd python3 "${ROOT}/scripts/summarize_dataset_quality.py" \
      --run-dir "${RUN_DIR}" \
      --quality-dir "${RUN_DIR}/quality" \
      --out-csv "${RUN_DIR}/summary/index_quality.csv"
  fi
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
  ln -sfn "${RUN_DIR}" "${ROOT}/runs/LATEST_${DATASET_NAME}_COMPARISON"
}

case "${MODE}" in
  all)
    require_data
    build_hidann
    build_baselines
    run_hidann_pipeline
    run_diskann_common_pq_construction
    run_sogaic_construction
    run_diskann_search
    run_pipeann_search
    run_same_index_search
    run_quality
    summarize
    plot
    mark_latest
    ;;
  build)
    build_hidann
    build_baselines
    ;;
  hidann) run_hidann_pipeline ;;
  diskann) run_diskann_common_pq_construction ;;
  sogaic) run_sogaic_construction ;;
  search) run_diskann_search ;;
  pipeann) run_pipeann_search ;;
  same_search|same-index-search) run_same_index_search ;;
  quality) run_quality; summarize; plot ;;
  summarize) summarize ;;
  plots) plot ;;
  -h|--help|help) usage ;;
  *) usage; exit 2 ;;
esac

echo
echo "${DATASET_LABEL} comparison outputs: ${RUN_DIR}"

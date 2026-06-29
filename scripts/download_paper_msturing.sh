#!/usr/bin/env bash
# Download the MSTuring dataset used by the HiDANN paper experiments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIG_ANN_DIR="${BIG_ANN_DIR:-$(dirname "${ROOT}")/big-ann-benchmarks}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/data/msturing}"
DATASET="${1:-msturing10M}"

if [[ "${DATASET}" != "msturing10M" ]]; then
  echo "unsupported MSTuring paper dataset: ${DATASET}" >&2
  echo "allowed dataset: msturing10M" >&2
  exit 2
fi

if [[ ! -d "${BIG_ANN_DIR}" ]]; then
  git clone https://github.com/harsha-simhadri/big-ann-benchmarks "${BIG_ANN_DIR}"
fi

PYTHON="${BIG_ANN_DIR}/.venv/bin/python3"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

(
  cd "${BIG_ANN_DIR}"
  "${PYTHON}" - <<'PY'
from benchmark.datasets import DATASETS

ds = DATASETS["msturing-10M"]()
ds.prepare(False)
PY
)

mkdir -p "${OUT_ROOT}/msturing10M"

BIG_ANN_DATA="${BIG_ANN_DIR}/data/MSTuringANNS"
link_file() {
  local src="$1"
  local dst="$2"
  if [[ ! -s "${src}" ]]; then
    echo "missing expected file: ${src}" >&2
    exit 1
  fi
  ln -sf "${src}" "${dst}"
  echo "linked ${dst} -> ${src}"
}

link_first_existing() {
  local dst="$1"
  shift
  for src in "$@"; do
    if [[ -s "${src}" ]]; then
      ln -sf "${src}" "${dst}"
      echo "linked ${dst} -> ${src}"
      return 0
    fi
  done
  echo "missing expected file for ${dst}" >&2
  printf '  tried: %s\n' "$@" >&2
  exit 1
}

link_file "${BIG_ANN_DATA}/base1b.fbin.crop_nb_10000000" "${OUT_ROOT}/msturing10M/msturing10M.fbin"
link_file "${BIG_ANN_DATA}/query100K.fbin" "${OUT_ROOT}/msturing10M/query100K.fbin"
link_first_existing \
  "${OUT_ROOT}/msturing10M/msturing10M_gt100" \
  "${BIG_ANN_DATA}/msturing-gt-10M" \
  "${BIG_ANN_DATA}/GT_10M/msturing-10M" \
  "${BIG_ANN_DATA}/gt10M.bin"

echo "MSTuring10M dataset is ready under ${OUT_ROOT}/msturing10M"

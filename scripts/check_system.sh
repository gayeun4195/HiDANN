#!/usr/bin/env bash
set -euo pipefail

missing=0

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2
    missing=1
  fi
}

need_file() {
  if [[ ! -e "$1" ]]; then
    echo "missing file: $1" >&2
    missing=1
  fi
}

path_has_eigen3() {
  local dir="$1"
  [[ -n "${dir}" ]] || return 1
  [[ -e "${dir}/Eigen/Core" || -e "${dir}/eigen3/Eigen/Core" ]]
}

path_list_has_eigen3() {
  local value="${1:-}"
  local normalized="${value//;/:}"
  local dir
  local dirs=()
  IFS=':' read -r -a dirs <<< "${normalized}"
  for dir in "${dirs[@]}"; do
    if path_has_eigen3 "${dir}"; then
      return 0
    fi
  done
  return 1
}

need_eigen3() {
  if path_has_eigen3 /usr/include/eigen3 || \
     path_has_eigen3 /usr/local/include/eigen3 || \
     path_has_eigen3 /usr/include || \
     path_has_eigen3 /usr/local/include || \
     path_list_has_eigen3 "${EIGEN3_INCLUDE_DIR:-}" || \
     path_list_has_eigen3 "${EIGEN3_INCLUDE_DIRS:-}" || \
     path_list_has_eigen3 "${CPATH:-}" || \
     path_list_has_eigen3 "${CPLUS_INCLUDE_PATH:-}"; then
    return
  fi

  echo "missing Eigen3 headers: install libeigen3-dev or set EIGEN3_INCLUDE_DIR/CPLUS_INCLUDE_PATH" >&2
  missing=1
}

have_pybind11_cmake() {
  local cmake_dir
  if [[ -n "${pybind11_DIR:-}" ]]; then
    [[ -e "${pybind11_DIR}/pybind11Config.cmake" || -e "${pybind11_DIR}/pybind11-config.cmake" ]] && return 0
  fi
  if [[ -n "${PYBIND11_DIR:-}" ]]; then
    [[ -e "${PYBIND11_DIR}/pybind11Config.cmake" || -e "${PYBIND11_DIR}/pybind11-config.cmake" ]] && return 0
  fi
  if cmake_dir="$(python3 -m pybind11 --cmakedir 2>/dev/null)" && [[ -n "${cmake_dir}" ]]; then
    [[ -e "${cmake_dir}/pybind11Config.cmake" || -e "${cmake_dir}/pybind11-config.cmake" ]] && return 0
  fi
  return 1
}

need_pybind11() {
  if [[ -e /usr/include/pybind11/pybind11.h ]] || have_pybind11_cmake; then
    return
  fi

  echo "missing pybind11: install pybind11-dev or install pybind11 in the active venv" >&2
  missing=1
}

need_cmd cmake
need_cmd g++
need_cmd git
need_cmd python3

need_python_module() {
  if ! python3 - "$1" <<'PY' >/dev/null 2>&1
import importlib
import sys

importlib.import_module(sys.argv[1])
PY
  then
    echo "missing Python package: $1" >&2
    missing=1
  fi
}

need_python_module numpy
need_python_module h5py

need_file /usr/include/boost/program_options.hpp
need_file /usr/include/gperftools/malloc_extension.h
need_file /usr/include/mkl/mkl.h
need_eigen3
need_pybind11

if ! ldconfig -p 2>/dev/null | grep -q 'libmkl_core.so'; then
  echo "missing runtime library: libmkl_core.so" >&2
  missing=1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libiomp5.so' && \
   [[ ! -e /usr/lib/x86_64-linux-gnu/libiomp5.so ]] && \
   [[ ! -e /opt/intel/oneapi/mkl/latest/lib/libiomp5.so ]] && \
   [[ ! -e /opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so ]]; then
  echo "missing runtime library: libiomp5.so" >&2
  missing=1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libaio.so'; then
  echo "missing runtime library: libaio.so" >&2
  missing=1
fi

if [[ -r /proc/sys/fs/aio-max-nr ]]; then
  aio_max="$(cat /proc/sys/fs/aio-max-nr)"
  if [[ "${aio_max}" =~ ^[0-9]+$ ]] && [[ "${aio_max}" -lt 131072 ]]; then
    cat >&2 <<EOF
warning: /proc/sys/fs/aio-max-nr=${aio_max}; 128-thread search may need at least 131072 libaio events.
         The run scripts cap SEARCH_THREADS automatically unless HIDANN_AIO_AUTO_CAP=0.
         To keep 128 search threads, run: sudo sysctl fs.aio-max-nr=1048576
EOF
  fi
fi

if [[ "$missing" != 0 ]]; then
  cat >&2 <<'EOF'

Install the common Ubuntu dependencies with:
  sudo apt update
  sudo apt install -y build-essential cmake git python3 python3-venv \
    python3-dev pybind11-dev libeigen3-dev \
    libboost-program-options-dev libgoogle-perftools-dev \
    libaio-dev liburing-dev libmkl-full-dev

Install the Python packages in the environment used to run the scripts with:
  python3 -m venv .venv
  . .venv/bin/activate
  python3 -m pip install numpy h5py pybind11

If pybind11 or Eigen are installed outside the system include paths, use:
  export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
  export EIGEN3_INCLUDE_DIR=/path/containing/Eigen/Core

If MKL is installed outside the system paths, configure CMake with:
  -DMKL_PATH=/path/to/mkl/lib -DMKL_INCLUDE_PATH=/path/to/mkl/include
EOF
  exit 1
fi

echo "system check passed"

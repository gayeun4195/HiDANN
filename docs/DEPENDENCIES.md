# Dependencies

The artifact is tested on Linux with GCC, CMake, OpenMP, Boost program_options,
Eigen3, pybind11, gperftools/tcmalloc, Linux AIO, and Intel MKL.

On Ubuntu, install the common packages with:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv \
  python3-dev pybind11-dev libeigen3-dev \
  libboost-program-options-dev libgoogle-perftools-dev \
  libaio-dev liburing-dev libmkl-full-dev
```

The VIBE downloader/converter also needs NumPy and h5py. On distributions that
protect the system Python environment, use a virtual environment:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install numpy h5py pybind11
```

The plotting script writes SVG directly with the Python standard library; it
does not require Matplotlib.

Run:

```bash
scripts/check_system.sh
```

The dependency check accepts either the system `pybind11-dev`/`libeigen3-dev`
packages or an active-venv pybind11 CMake package plus an Eigen include path:

```bash
export pybind11_DIR="$(python3 -m pybind11 --cmakedir)"
export EIGEN3_INCLUDE_DIR=/path/containing/Eigen/Core
```

DiskANN-style search opens one libaio queue per search worker. The artifact
wrappers preflight `/proc/sys/fs/aio-max-nr` and cap `SEARCH_THREADS`
automatically when the host limit is too small. To keep 128 search threads:

```bash
sudo sysctl fs.aio-max-nr=1048576
```

Set `HIDANN_AIO_AUTO_CAP=0` to turn the automatic cap into a hard preflight
error.

If MKL is installed outside the default system paths, pass the library and
header directories to CMake:

```bash
CMAKE_EXTRA_ARGS="-DMKL_PATH=/path/to/mkl/lib -DMKL_INCLUDE_PATH=/path/to/mkl/include" \
  scripts/run_artifact_pipeline.sh build
```

The C++ binaries are linked against the MKL ILP64 runtime. On the pre-submission
test machine, `ldd` resolved these relevant shared libraries:

- `libmkl_intel_ilp64.so`
- `libmkl_intel_thread.so`
- `libmkl_core.so`
- `libomp.so.5`
- `libtcmalloc.so.4`
- `libaio.so` for the search binary

For the full paper experiments, use a Release build and a machine with enough
memory and storage for the target dataset. The formal server runs used 128
threads; the artifact smoke uses fewer threads by default so it can run quickly.

The run scripts can use user-scoped `systemd-run` to apply cgroup-style memory
caps. If
`systemd-run --user --scope` is unavailable, set `USE_CGROUP=0`; the script will
still run and will report `/usr/bin/time -v` RSS, but the OS will not enforce
the memory cap.

The bundled PipeANN baseline is built with `-DUSE_AIO=ON` in the artifact
wrapper. This uses Linux AIO (`libaio-dev`) and avoids environment-specific
issues with the vendored `io_uring` headers while preserving the same
`tests/search_disk_index` interface used by the search baseline.

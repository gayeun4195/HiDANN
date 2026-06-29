mkdir -p build
cd build
PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
cmake -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}" ..
make -j

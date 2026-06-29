cmake_minimum_required(VERSION 3.18)

# Keep the historical target name "diskann" because the HiDANN source files
# and subdirectory CMake files link against that core library target.
project(diskann LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PORTABLE "Disable -march=native for artifact builds" ON)
option(USE_LIKWID "Enable LIKWID markers" OFF)

include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenMP REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options)

find_path(GPERFTOOLS_INCLUDE_DIR gperftools/malloc_extension.h)
if(NOT GPERFTOOLS_INCLUDE_DIR)
  message(FATAL_ERROR "Missing gperftools headers. Install libgoogle-perftools-dev.")
endif()
include_directories(${GPERFTOOLS_INCLUDE_DIR})

find_library(TCMALLOC_LIB tcmalloc)
if(TCMALLOC_LIB)
  set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS ${TCMALLOC_LIB})
  message(STATUS "Using tcmalloc: ${TCMALLOC_LIB}")
else()
  set(DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS "")
  message(WARNING "tcmalloc library not found; continuing without tcmalloc link option.")
endif()

find_library(AIO_LIB aio REQUIRED)
set(DISKANN_ASYNC_LIB ${AIO_LIB})
find_library(URING_LIB uring)
if(URING_LIB)
  list(APPEND DISKANN_ASYNC_LIB ${URING_LIB})
  add_compile_definitions(SIGMA_WITH_IO_URING)
  message(STATUS "Using liburing: ${URING_LIB}")
else()
  message(STATUS "liburing not found; Linux AIO path remains enabled.")
endif()

if(USE_LIKWID)
  find_path(LIKWID_INCLUDE_DIR likwid.h)
  find_library(LIKWID_LIB likwid)
  if(NOT LIKWID_INCLUDE_DIR OR NOT LIKWID_LIB)
    message(FATAL_ERROR "USE_LIKWID=ON but likwid was not found.")
  endif()
  include_directories(${LIKWID_INCLUDE_DIR})
  link_libraries(${LIKWID_LIB})
  add_compile_definitions(LIKWID_PERFMON)
endif()

set(MKL_PATH "" CACHE PATH "Directory containing Intel MKL shared libraries")
set(MKL_INCLUDE_PATH "" CACHE PATH "Directory containing mkl.h")

if(NOT MKL_PATH)
  foreach(candidate
      /opt/intel/oneapi/mkl/latest/lib/intel64
      /opt/intel/oneapi/mkl/latest/lib
      /usr/lib/x86_64-linux-gnu
      /opt/intel/mkl/lib/intel64)
    if(EXISTS "${candidate}/libmkl_core.so")
      set(MKL_PATH "${candidate}")
      break()
    endif()
  endforeach()
endif()

if(NOT MKL_INCLUDE_PATH)
  foreach(candidate
      /opt/intel/oneapi/mkl/latest/include
      /usr/include/mkl
      /opt/intel/mkl/include)
    if(EXISTS "${candidate}/mkl.h")
      set(MKL_INCLUDE_PATH "${candidate}")
      break()
    endif()
  endforeach()
endif()

if(NOT MKL_PATH OR NOT MKL_INCLUDE_PATH)
  message(FATAL_ERROR
    "Intel MKL was not found. Pass -DMKL_PATH=/path/to/mkl/lib "
    "and -DMKL_INCLUDE_PATH=/path/to/mkl/include, or install libmkl-full-dev.")
endif()

find_library(MKL_INTEL_ILP64 mkl_intel_ilp64 HINTS ${MKL_PATH} REQUIRED)
find_library(MKL_INTEL_THREAD mkl_intel_thread HINTS ${MKL_PATH} REQUIRED)
find_library(MKL_CORE mkl_core HINTS ${MKL_PATH} REQUIRED)
find_library(IOMP5 iomp5 HINTS ${MKL_PATH} /usr/lib/x86_64-linux-gnu REQUIRED)

include_directories(${MKL_INCLUDE_PATH})
link_directories(${MKL_PATH})
link_libraries(${MKL_INTEL_ILP64} ${MKL_INTEL_THREAD} ${MKL_CORE} ${IOMP5} pthread m dl)
link_libraries(OpenMP::OpenMP_CXX)

add_compile_definitions(MKL_ILP64 USE_AVX2)
add_compile_options(-mavx2 -mfma -msse2 -ftree-vectorize -fopenmp -fopenmp-simd -funroll-loops)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(-DNDEBUG -Ofast)
  if(NOT PORTABLE)
    add_compile_options(-march=native -mtune=native)
  endif()
endif()

add_subdirectory(src)
add_subdirectory(construction)
add_subdirectory(search)
add_subdirectory(tools)

set(CMAKE_CXX_STANDARD 17)
add_definitions(-DREAD_ONLY_TESTS -DNO_MAPPING)

file(GLOB PROJECT_CPP *.cpp search/*.cpp update/*.cpp utils/*.cpp)
add_library(${PROJECT_NAME} ${PROJECT_CPP})

# python module.
# pip install "pybind11[global]"
find_package(pybind11 REQUIRED)
file(GLOB PYBIND_CPP python/*.cpp)
pybind11_add_module(C ${PYBIND_CPP})
target_link_libraries(C PRIVATE pipeann)

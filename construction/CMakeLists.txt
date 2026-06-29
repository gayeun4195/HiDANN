# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

set(CMAKE_CXX_STANDARD 17)

add_executable(our_construction our_construction.cpp)
target_link_libraries(our_construction ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(our_pruning our_pruning.cpp)
target_link_libraries(our_pruning ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

add_executable(reverse_edge_logic reverse_edge_logic.cpp)
target_link_libraries(reverse_edge_logic ${PROJECT_NAME} ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS} Boost::program_options)

if (NOT MSVC)
    install(TARGETS our_construction our_pruning reverse_edge_logic RUNTIME)
endif()

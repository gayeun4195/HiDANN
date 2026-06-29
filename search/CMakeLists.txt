# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

if (NOT MSVC)
    include_directories(${PROJECT_SOURCE_DIR}/search)
    include_directories(${PROJECT_SOURCE_DIR}/include)

    # Compile the new search logic into a static library
    add_library(ours_lib STATIC ours_search.cpp ours_data_interfaces.cpp)
    target_link_libraries(ours_lib diskann ${DISKANN_ASYNC_LIB} Boost::program_options ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

    # Search Executable
    add_executable(ours_search ours_search_main.cpp)
    target_link_libraries(ours_search ours_lib diskann ${DISKANN_ASYNC_LIB} Boost::program_options ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

    # Cache Generator Executable
    add_executable(ours_cache_gen ours_cache_gen.cpp)
    target_link_libraries(ours_cache_gen ours_lib diskann ${DISKANN_ASYNC_LIB} Boost::program_options ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS})

    install(TARGETS ours_search ours_cache_gen
            RUNTIME DESTINATION bin)
endif()

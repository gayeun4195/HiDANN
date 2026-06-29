// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <index.h>
#include <math_utils.h>
#include "sogaic_partition.h"

int main(int argc, char **argv)
{
    if (argc < 9)
    {
        std::cout << "Usage:\n"
                  << argv[0]
                  << "  datatype<int8/uint8/float>  <data_path>"
                     "  <prefix_path>  <sampling_rate>  "
                     "  <budget_ratio> <graph_degree_R> "
                     "  <max_overlap> <epsilon> [gamma_slack (default 1.0)]"
                     "  [chunk_bytes] [memory_budget_bytes] [partition_budget_fraction]"
                  << std::endl;
        exit(-1);
    }

    const std::string data_path(argv[2]);
    const std::string prefix_path(argv[3]);
    const float sampling_rate = (float)atof(argv[4]);
    const float budget_ratio = (float)atof(argv[5]);
    const size_t graph_degree = (size_t)std::atoi(argv[6]);
    const float max_overlap = (float)atof(argv[7]);
    const float epsilon = (float)atof(argv[8]);
    const float gamma_slack = argc > 9 ? (float)atof(argv[9]) : 1.0f;
    const uint64_t chunk_bytes = argc > 10 ? (uint64_t)std::stoull(argv[10]) : 0;
    const uint64_t memory_budget_bytes = argc > 11 ? (uint64_t)std::stoull(argv[11]) : 0;
    const float partition_budget_fraction = argc > 12 ? (float)atof(argv[12]) : 1.0f;

    if (chunk_bytes > 0)
    {
        std::string s = std::to_string(chunk_bytes);
        setenv("DISKANN_CHUNK_BYTES", s.c_str(), 1);
        std::cout << "[sogaic_partition_data] Applied DISKANN_CHUNK_BYTES=" << s << " from argv[10]\n";
    }
    
    const size_t max_reps = 15;

    std::cout << "Running SOGAIC Partitioning..." << std::endl;
    std::cout << "Data: " << data_path << std::endl;
    std::cout << "Target Budget Ratio: " << budget_ratio << std::endl;
    std::cout << "Graph Degree R: " << graph_degree << std::endl;
    std::cout << "Max Overlap (Omega): " << max_overlap << std::endl;
    std::cout << "Epsilon: " << epsilon << std::endl;
    std::cout << "Memory Budget Bytes: " << memory_budget_bytes << std::endl;
    std::cout << "Partition Budget Fraction: " << partition_budget_fraction << std::endl;

    int ret_val = -1;
    if (std::string(argv[1]) == std::string("float"))
        ret_val = sogaic_partition<float>(data_path, sampling_rate, budget_ratio, graph_degree, max_reps, prefix_path, max_overlap, epsilon, gamma_slack, memory_budget_bytes, partition_budget_fraction);
    else if (std::string(argv[1]) == std::string("int8"))
        ret_val = sogaic_partition<int8_t>(data_path, sampling_rate, budget_ratio, graph_degree, max_reps, prefix_path, max_overlap, epsilon, gamma_slack, memory_budget_bytes, partition_budget_fraction);
    else if (std::string(argv[1]) == std::string("uint8"))
        ret_val = sogaic_partition<uint8_t>(data_path, sampling_rate, budget_ratio, graph_degree, max_reps, prefix_path, max_overlap, epsilon, gamma_slack, memory_budget_bytes, partition_budget_fraction);
    else
    {
        std::cout << "unsupported data format. use float/int8/uint8" << std::endl;
        return -1;
    }

    return ret_val;
}

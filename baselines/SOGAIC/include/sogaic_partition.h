// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#include <cstdint>

#include "partition.h"

template <typename T>
int sogaic_partition(const std::string data_file, const float sampling_rate, float budget_ratio, size_t graph_degree, size_t max_k_means_reps,
                     const std::string prefix_path, float max_overlap_factor, float epsilon, float gamma_slack,
                     uint64_t memory_budget_bytes = 0, float partition_budget_fraction = 1.0f);

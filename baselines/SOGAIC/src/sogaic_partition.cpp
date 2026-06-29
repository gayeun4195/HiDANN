// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cmath>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdint>
#include <limits>

#include "utils.h"
#include "math_utils.h"
#include "index.h"
#include "parameters.h"
#include "memory_mapper.h"
#include "partition.h"
#include "sogaic_partition.h"

// BLOCK_SIZE is now calculated dynamically
// #define BLOCK_SIZE 100000

// Helper struct for sorting distances
struct CentroidDist {
    size_t id;
    float dist;
    bool operator<(const CentroidDist& other) const {
        return dist < other.dist;
    }
};

namespace
{
constexpr double BYTES_IN_GIB = 1024.0 * 1024.0 * 1024.0;

template <typename T>
uint64_t derive_logical_memory_budget_bytes(size_t npts, size_t dim, size_t graph_degree, float budget_ratio)
{
    const double data_bytes = 8.0 + static_cast<double>(npts) * static_cast<double>(dim) * sizeof(T);
    const double graph_bytes = static_cast<double>(npts) * static_cast<double>(graph_degree) * sizeof(uint32_t);
    const double budget_bytes = (data_bytes + graph_bytes) * static_cast<double>(budget_ratio);

    if (budget_bytes <= 0.0)
    {
        return 0;
    }
    if (budget_bytes >= static_cast<double>(std::numeric_limits<uint64_t>::max()))
    {
        return std::numeric_limits<uint64_t>::max();
    }
    return static_cast<uint64_t>(budget_bytes);
}

template <typename T>
size_t max_capacity_for_memory_budget(uint64_t memory_budget_bytes, size_t dim, size_t graph_degree, size_t npts)
{
    if (memory_budget_bytes == 0 || npts == 0)
    {
        return 0;
    }

    size_t low = 1;
    size_t high = npts;
    size_t best = 0;
    while (low <= high)
    {
        const size_t mid = low + (high - low) / 2;
        const double estimated_ram =
            diskann::estimate_ram_usage(mid, static_cast<uint32_t>(dim), sizeof(T), static_cast<uint32_t>(graph_degree));

        if (estimated_ram <= static_cast<double>(memory_budget_bytes))
        {
            best = mid;
            low = mid + 1;
        }
        else
        {
            if (mid == 0)
            {
                break;
            }
            high = mid - 1;
        }
    }

    return best;
}
} // namespace

template <typename T>
int sogaic_shard_data_into_clusters(const std::string data_file, float *pivots, const size_t num_centers, const size_t dim,
                             float max_overlap_factor, float epsilon, size_t max_capacity, size_t graph_degree,
                             std::string prefix_path, size_t target_ram_bytes)
{
    // ... (reader setup) ...
    size_t read_blk_size = 64 * 1024 * 1024;
    cached_ifstream base_reader(data_file, read_blk_size);
    uint32_t npts32;
    uint32_t basedim32;
    base_reader.read((char *)&npts32, sizeof(uint32_t));
    base_reader.read((char *)&basedim32, sizeof(uint32_t));
    size_t num_points = npts32;
    if (basedim32 != dim)
    {
        diskann::cout << "Error. dimensions dont match for train set and base set" << std::endl;
        return -1;
    }

    std::vector<size_t> shard_counts(num_centers, 0);
    const size_t max_overlap_count = static_cast<size_t>(max_overlap_factor);
    if (max_overlap_count == 0)
    {
        diskann::cerr << "Error: max_overlap_factor must be at least 1. Got: " << max_overlap_factor << std::endl;
        return -1;
    }
    size_t points_with_assignment = 0;
    size_t unassigned_count = 0;
    std::vector<std::ofstream> shard_data_writer(num_centers);
    std::vector<std::ofstream> shard_idmap_writer(num_centers);
    uint32_t dummy_size = 0;
    uint32_t const_one = 1;

    // Initialize writers
    for (size_t i = 0; i < num_centers; i++)
    {
        std::string data_filename = prefix_path + "_subshard-" + std::to_string(i) + ".bin";
        std::string idmap_filename = prefix_path + "_subshard-" + std::to_string(i) + "_ids_uint32.bin";
        shard_data_writer[i] = std::ofstream(data_filename.c_str(), std::ios::binary);
        shard_idmap_writer[i] = std::ofstream(idmap_filename.c_str(), std::ios::binary);
        // Write placeholders for size and dim
        shard_data_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_data_writer[i].write((char *)&basedim32, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&dummy_size, sizeof(uint32_t));
        shard_idmap_writer[i].write((char *)&const_one, sizeof(uint32_t));
    }


    // Dynamic Block Size Calculation
    // Priority: DISKANN_CHUNK_BYTES (Env) > Default (100000 - Original)
    // Note: Previously we used target_ram_bytes passed from budget_ratio, but shifting to Env Var control for consistency.
    
    size_t block_size = 100000; // Default original
    if (const char* env_p = std::getenv("DISKANN_CHUNK_BYTES")) {
        try {
            size_t env_target = std::stoull(env_p);
            size_t bytes_per_point = dim * sizeof(T) + dim * sizeof(float) + num_centers * sizeof(float) + sizeof(float);
            size_t calculated_size = env_target / bytes_per_point;
            if (calculated_size < 1000) calculated_size = 1000;
            if (calculated_size > 5000000) calculated_size = 5000000; // Cap at 5M
            block_size = calculated_size;
            diskann::cout << "Dynamic Block Size: " << block_size << " (Target Budget: " << env_target / 1024 / 1024 << "MB, Bytes/Pt: " << bytes_per_point << ")" << std::endl;
        } catch (...) {}
    } else {
         // Fallback: If budget_ratio logic was preferred, scripts should export DISKANN_CHUNK_BYTES.
         // Otherwise, we default to fixed size.
         // diskann::cout << "Using Default Block Size: " << block_size << std::endl;
    }
    
    // Ensure we don't exceed point count
    if (block_size > num_points) block_size = num_points;

    
    // Scratch space for block processing
    std::unique_ptr<T[]> block_data_T = std::make_unique<T[]>(block_size * dim);
    std::unique_ptr<float[]> block_data_float = std::make_unique<float[]>(block_size * dim);
    
    // Precompute pivot norms
    std::unique_ptr<float[]> pivs_norms_squared = std::make_unique<float[]>(num_centers);
    math_utils::compute_vecs_l2sq(pivs_norms_squared.get(), pivots, num_centers, dim);

    // Block norms and distance matrix
    std::unique_ptr<float[]> pts_norms_squared = std::make_unique<float[]>(block_size);
    std::unique_ptr<float[]> dist_matrix = std::make_unique<float[]>(block_size * num_centers);
    
    // Dummy index buffer for compute_closest_centers_in_block (we ignore it but must provide it)
    std::unique_ptr<uint32_t[]> dummy_center_index = std::make_unique<uint32_t[]>(block_size * 1); // k=1

    size_t num_blocks = DIV_ROUND_UP(num_points, block_size);

    diskann::cout << "SOGAIC Partitioning: N=" << num_points << ", P=" << num_centers 
                  << ", Omega=" << max_overlap_factor << ", Epsilon=" << epsilon << ", Gamma=" << max_capacity << std::endl;

    for (size_t block = 0; block < num_blocks; block++)
    {
        size_t start_id = block * block_size;
        size_t end_id = (std::min)((block + 1) * block_size, num_points);
        size_t cur_blk_size = end_id - start_id;

        // Read block
        base_reader.read((char *)block_data_T.get(), sizeof(T) * (cur_blk_size * dim));
        diskann::convert_types<T, float>(block_data_T.get(), block_data_float.get(), cur_blk_size, dim);

        // Compute norms for points
        math_utils::compute_vecs_l2sq(pts_norms_squared.get(), block_data_float.get(), cur_blk_size, dim);

        // Compute full distance matrix (N_block x Num_centers)
        // using existing math_utils function which wraps cblas
        math_utils::compute_closest_centers_in_block(
            block_data_float.get(), cur_blk_size, dim,
            pivots, num_centers,
            pts_norms_squared.get(), pivs_norms_squared.get(),
            dummy_center_index.get(), dist_matrix.get(), 1 // k=1, result ignored
        );

        // Iterate over points in block (Sequential for strict capacity adherence)
        for (size_t p = 0; p < cur_blk_size; p++)
        {
            // Gather distances for filtering/sorting
            // dist_matrix is row-major: [p * num_centers + c]
            std::vector<CentroidDist> dists(num_centers);
            for(size_t c = 0; c < num_centers; c++) {
                dists[c] = {c, dist_matrix[p * num_centers + c]};
            }

            // Algorithm 1: Adaptive Overload-Aware Vector Assignment
            // Sort by distance (asc) - PriorityQueue implementation
            std::sort(dists.begin(), dists.end());

            size_t curOLPCnt = 0;
            size_t curOLPFactor = 0;
            float accDist = 0.0f;
            float curAVGDist = std::numeric_limits<float>::max(); // Infinity
            bool assigned_to_any_shard = false;

            // Greedy assignment
            for(const auto& item : dists) {
                size_t c_idx = item.id;
                float d = item.dist;

                if (curOLPCnt >= max_overlap_count) break;

                // Condition: dist <= epsilon * curAVGDist
                // Note: For the first item, curAVGDist is INF, so condition is true.
                if (d <= epsilon * curAVGDist) {
                    curOLPFactor++;
                    accDist += d;
                    
                    // Check capacity
                    if (shard_counts[c_idx] < max_capacity) {
                        curOLPCnt++;
                        
                        // Assign to shard
                        uint32_t original_point_map_id = (uint32_t)(start_id + p);
                        shard_data_writer[c_idx].write((char *)(block_data_T.get() + p * dim), sizeof(T) * dim);
                        shard_idmap_writer[c_idx].write((char *)&original_point_map_id, sizeof(uint32_t));
                        
                        shard_counts[c_idx]++;
                        assigned_to_any_shard = true;
                        
                        // Update AVG (only if assigned or just if considered 'close enough'?
                        // Paper Line 10-12: Update stats BEFORE checking capacity.
                        // "curOLPFactor++ ... curAVGDist = ..."
                        // Then "if |S| < Gamma ... else curAVGDist = INF".
                        // So we update stats first.
                        curAVGDist = accDist / curOLPFactor;
                    } else {
                        // Overload!
                        // Paper Line 17: curAVGDist <- infinity
                        curAVGDist = std::numeric_limits<float>::max();
                        // Note: We do NOT increment curOLPCnt here.
                        // We do NOT add to shard.
                        // We just reset AVG to allow next centroid (which is farther) to be considered 'close enough' relative to 'infinity'.
                    }
                }
            }
            
            // Do not silently drop points or force them beyond Gamma. If this is
            // nonzero, the chosen Gamma/P/assignment combination violates the
            // SOGAIC overload-aware assignment invariant.
            if (assigned_to_any_shard) {
                points_with_assignment++;
            } else {
                unassigned_count++;
                if (unassigned_count <= 10) {
                    diskann::cout << "Warning: point " << (start_id + p)
                                  << " was not assigned to any non-full shard." << std::endl;
                }
            }
        }
        if (block % 10 == 0) {
            diskann::cout << "Processed block " << block << "/" << num_blocks << "...\r" << std::flush;
        }
    }
    diskann::cout << std::endl;

    // Finalize writers
    size_t total_count = 0;
    size_t max_shard_count = 0;
    size_t overloaded_shards = 0;
    diskann::cout << "SOGAIC Partitioning Result:" << std::endl;
    for (size_t i = 0; i < num_centers; i++)
    {
        uint32_t cur_shard_count = (uint32_t)shard_counts[i];
        total_count += cur_shard_count;
        max_shard_count = (std::max)(max_shard_count, shard_counts[i]);
        if (shard_counts[i] > max_capacity)
        {
            overloaded_shards++;
        }
        diskann::cout << "Shard " << i << ": " << cur_shard_count << std::endl;
        
        shard_data_writer[i].seekp(0);
        shard_data_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_data_writer[i].close();
        
        shard_idmap_writer[i].seekp(0);
        shard_idmap_writer[i].write((char *)&cur_shard_count, sizeof(uint32_t));
        shard_idmap_writer[i].close();
    }

    diskann::cout << "Total assigned points: " << total_count << " (Avg overlap: " << (float)total_count / num_points << ")" << std::endl;
    diskann::cout << "Unique points assigned at least once: " << points_with_assignment << "/" << num_points << std::endl;
    diskann::cout << "Unassigned points: " << unassigned_count << std::endl;
    diskann::cout << "Max shard size: " << max_shard_count << " / Gamma " << max_capacity << std::endl;
    diskann::cout << "Estimated RAM for max shard: "
                  << diskann::estimate_ram_usage(max_shard_count, static_cast<uint32_t>(dim), sizeof(T),
                                                 static_cast<uint32_t>(graph_degree))
                  << " bytes" << std::endl;

    if (unassigned_count != 0 || points_with_assignment != num_points)
    {
        diskann::cerr << "Error: SOGAIC assignment left " << unassigned_count
                      << " points without any shard assignment." << std::endl;
        return -1;
    }
    if (overloaded_shards != 0)
    {
        diskann::cerr << "Error: SOGAIC assignment produced " << overloaded_shards
                      << " shards above Gamma." << std::endl;
        return -1;
    }
    return 0;
}

template <typename T>
int sogaic_partition(const std::string data_file, const float sampling_rate, float budget_ratio, size_t graph_degree, size_t max_k_means_reps,
                     const std::string prefix_path, float max_overlap_factor, float epsilon, float gamma_ratio,
                     uint64_t memory_budget_bytes, float partition_budget_fraction)
{
    size_t train_dim;
    size_t num_train;
    float *train_data_float;

    // 1. Read metadata to determine N and Dim
    size_t npts;
    size_t dim;
    {
        cached_ifstream base_reader(data_file, 1024);
        uint32_t npts32;
        uint32_t dim32;
        base_reader.read((char *)&npts32, sizeof(uint32_t));
        base_reader.read((char *)&dim32, sizeof(uint32_t));
        npts = npts32;
        dim = dim32;
        train_dim = dim; // Assuming same dim for now
    }

    // 2. Calculate Gamma (Capacity) & Num Partitions (P) from the per-container
    // memory budget. In the SOGAIC paper, Gamma is the maximum number of base
    // points a division can hold under the container memory limit.
    
    if (budget_ratio <= 0.0f || budget_ratio > 1.0f) {
        diskann::cerr << "Error: Budget ratio must be between 0.0 and 1.0. Got: " << budget_ratio << std::endl;
        return -1;
    }
    if (max_overlap_factor < 1.0f) {
        diskann::cerr << "Error: Max overlap factor must be at least 1. Got: " << max_overlap_factor << std::endl;
        return -1;
    }
    if (partition_budget_fraction <= 0.0f || partition_budget_fraction > 1.0f) {
        diskann::cerr << "Error: Partition budget fraction must be between 0.0 and 1.0. Got: "
                      << partition_budget_fraction << std::endl;
        return -1;
    }

    uint64_t effective_memory_budget_bytes = memory_budget_bytes;
    if (effective_memory_budget_bytes == 0) {
        effective_memory_budget_bytes = derive_logical_memory_budget_bytes<T>(npts, dim, graph_degree, budget_ratio);
        diskann::cout << "Warning: memory_budget_bytes was not provided; derived a logical budget from "
                      << "(data_bytes + graph_bytes) * budget_ratio." << std::endl;
    }
    const uint64_t capacity_budget_bytes =
        static_cast<uint64_t>(static_cast<long double>(effective_memory_budget_bytes) *
                              static_cast<long double>(partition_budget_fraction));
    if (capacity_budget_bytes == 0) {
        diskann::cerr << "Error: Effective partition capacity budget is zero. Memory budget bytes: "
                      << effective_memory_budget_bytes << ", partition budget fraction: "
                      << partition_budget_fraction << std::endl;
        return -1;
    }

    size_t max_capacity = max_capacity_for_memory_budget<T>(capacity_budget_bytes, dim, graph_degree, npts);
    if (max_capacity == 0) {
        diskann::cerr << "Error: Memory budget is too small to fit even one vector under estimate_ram_usage(). Budget bytes: "
                      << capacity_budget_bytes << std::endl;
        return -1;
    }
    const double estimated_ram_at_gamma =
        diskann::estimate_ram_usage(max_capacity, static_cast<uint32_t>(dim), sizeof(T), static_cast<uint32_t>(graph_degree));
    const double estimated_full_ram =
        diskann::estimate_ram_usage(npts, static_cast<uint32_t>(dim), sizeof(T), static_cast<uint32_t>(graph_degree));
    
    // P = ceil(Omega * N / Gamma)
    size_t num_centers = (size_t)std::ceil((double)max_overlap_factor * npts / max_capacity);
    if (num_centers < 1) num_centers = 1;

    diskann::cout << "SOGAIC Configuration Analysis:" << std::endl;
    diskann::cout << "  N: " << npts << ", Dim: " << dim << std::endl;
    diskann::cout << "  Budget Ratio: " << budget_ratio * 100.0f << "%" << std::endl;
    diskann::cout << "  Memory Budget: " << effective_memory_budget_bytes << " bytes ("
                  << (static_cast<double>(effective_memory_budget_bytes) / BYTES_IN_GIB) << " GiB)" << std::endl;
    diskann::cout << "  Partition Budget Fraction: " << partition_budget_fraction << std::endl;
    diskann::cout << "  Capacity Budget: " << capacity_budget_bytes << " bytes ("
                  << (static_cast<double>(capacity_budget_bytes) / BYTES_IN_GIB) << " GiB)" << std::endl;
    diskann::cout << "  Estimated Full Build RAM: " << estimated_full_ram << " bytes ("
                  << (estimated_full_ram / BYTES_IN_GIB) << " GiB)" << std::endl;
    diskann::cout << "  Calculated Capacity (Gamma): " << max_capacity << " vectors" << std::endl;
    diskann::cout << "  Estimated RAM at Gamma: " << estimated_ram_at_gamma << " bytes ("
                  << (estimated_ram_at_gamma / BYTES_IN_GIB) << " GiB)" << std::endl;
    diskann::cout << "  Calculated Partitions (P): " << num_centers << std::endl;
    diskann::cout << "  Gamma slack argument (compatibility): " << gamma_ratio << std::endl;
    
    // Output P to file for script to read
    std::ofstream p_out(prefix_path + "_num_partitions.txt");
    p_out << num_centers;
    p_out.close();

    float final_sampling_rate = sampling_rate;
    if (final_sampling_rate <= 0.0f) {
        if (npts > 256000) {
            final_sampling_rate = 256000.0f / npts;
            diskann::cout << "  Auto Sampling Rate: " << final_sampling_rate << std::endl;
        } else {
            final_sampling_rate = 1.0f;
        }
    }

    // 3. Sample data for K-Means
    gen_random_slice<T>(data_file, final_sampling_rate, train_data_float, num_train, train_dim);

    float *pivot_data = new float[num_centers * train_dim];
    std::string output_file = prefix_path + "_centroids.bin";

    // 4. K-Means Clustering
    diskann::cout << "Processing global k-means with P=" << num_centers << "..." << std::endl;
    auto kmeans_start = std::chrono::high_resolution_clock::now();
    kmeans::kmeanspp_selecting_pivots(train_data_float, num_train, train_dim, pivot_data, num_centers);
    kmeans::run_lloyds(train_data_float, num_train, train_dim, pivot_data, num_centers, max_k_means_reps, NULL, NULL);
    auto kmeans_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> kmeans_diff = kmeans_end - kmeans_start;
    diskann::cout << "K-Means Time: " << kmeans_diff.count() << "s" << std::endl;

    diskann::save_bin<float>(output_file.c_str(), pivot_data, (size_t)num_centers, train_dim);

    // 5. Sharding
    // We use the calculated max_capacity (Gamma)
    // Use 50% of the same logical budget for sharding scratch buffers.
    size_t target_sharding_ram = static_cast<size_t>(effective_memory_budget_bytes / 2);
    if (target_sharding_ram < 1024 * 1024) target_sharding_ram = 1024 * 1024; // Min 1MB

    auto sharding_start = std::chrono::high_resolution_clock::now();
    int shard_ret = sogaic_shard_data_into_clusters<T>(data_file, pivot_data, num_centers, train_dim,
                                                       max_overlap_factor, epsilon, max_capacity, graph_degree,
                                                       prefix_path, target_sharding_ram);
    auto sharding_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> sharding_diff = sharding_end - sharding_start;
    diskann::cout << "Assignment Time: " << sharding_diff.count() << "s" << std::endl;

    delete[] pivot_data;
    delete[] train_data_float;
    return shard_ret;
}

// Instantiate templates
template int sogaic_partition<int8_t>(const std::string data_file, const float sampling_rate, float budget_ratio, size_t graph_degree, size_t max_k_means_reps,
                                      const std::string prefix_path, float max_overlap_factor, float epsilon, float gamma_ratio,
                                      uint64_t memory_budget_bytes, float partition_budget_fraction);
template int sogaic_partition<uint8_t>(const std::string data_file, const float sampling_rate, float budget_ratio, size_t graph_degree, size_t max_k_means_reps,
                                       const std::string prefix_path, float max_overlap_factor, float epsilon, float gamma_ratio,
                                       uint64_t memory_budget_bytes, float partition_budget_fraction);
template int sogaic_partition<float>(const std::string data_file, const float sampling_rate, float budget_ratio, size_t graph_degree, size_t max_k_means_reps,
                                     const std::string prefix_path, float max_overlap_factor, float epsilon, float gamma_ratio,
                                     uint64_t memory_budget_bytes, float partition_budget_fraction);

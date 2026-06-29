// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <queue>
#include <unordered_set>
#include <numeric>
#include <omp.h>
#include <boost/program_options.hpp>
#include "tsl/robin_map.h"

#include "utils.h"
#include "ours_data_interfaces.h"
#include "linux_aligned_file_reader.h"

namespace po = boost::program_options;

// Helper struct for sorting by in-degree
struct NodeIndegree {
    uint32_t id;
    uint32_t degree;

    bool operator>(const NodeIndegree& other) const {
        if (degree != other.degree)
            return degree > other.degree; // Higher degree first
        return id < other.id; // Tie-break with ID
    }
};

int main(int argc, char** argv) {
    std::string graph_path; // Can be layout dependent prefix or direct file
    std::string output_file;
    uint32_t num_nodes_to_cache;
    uint32_t bfs_hops;
    bool use_fixed_layout;
    uint32_t R; // For fixed layout
    uint32_t start_node;

    std::string mode; // "hybrid" or "bfs"

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "produce help message")
        ("graph_path", po::value<std::string>(&graph_path)->required(), "Path to graph file (or prefix if using SIGMA layout)")
        ("output_file", po::value<std::string>(&output_file)->required(), "Output file path for cache list")
        ("num_nodes_to_cache,k", po::value<uint32_t>(&num_nodes_to_cache)->required(), "Total number of nodes to cache")
        ("start_node,s", po::value<uint32_t>(&start_node)->default_value(0), "Start node for BFS (Medoid)")
        ("bfs_hops", po::value<uint32_t>(&bfs_hops)->default_value(3), "Number of hops for initial BFS (Hybrid mode only)")
        ("mode", po::value<std::string>(&mode)->default_value("hybrid"), "Cache selection mode: 'hybrid' (BFS+Degree) or 'bfs' (Pure BFS)")
        ("fixed_layout", po::bool_switch(&use_fixed_layout), "Use SIGMA Fixed-R layout (assumes .bin suffix)")
        ("R", po::value<uint32_t>(&R)->default_value(70), "Max degree R (only for fixed layout)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) {
            std::cout << desc << "\n";
            return 0;
        }
        po::notify(vm);
    } catch(const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    if (mode != "hybrid" && mode != "bfs") {
        std::cerr << "Error: mode must be 'hybrid' or 'bfs'" << std::endl;
        return 1;
    }

    try {
    std::cout << "Starting Sigma Cache Generator..." << std::endl;
    std::cout << "Target Cache Size: " << num_nodes_to_cache << std::endl;
    std::cout << "Mode: " << mode << std::endl;

    // ------------------------------------------------------------------
    // 0. Auto-Detect Start Node (Medoid) if not provided
    // ------------------------------------------------------------------
    if (start_node == 0) {
        // Check for _medoids.bin
        std::string prefix = graph_path;
        if (prefix.size() > 11 && prefix.substr(prefix.size() - 11) == "_disk.index") {
            prefix = prefix.substr(0, prefix.size() - 11);
        }
        std::string medoids_file = prefix + "_medoids.bin";

        std::ifstream m_in(medoids_file, std::ios::binary);
        if (m_in.is_open()) {
            size_t n_medoids, dim_medoids;
            m_in.read((char*)&n_medoids, 4);
            m_in.read((char*)&dim_medoids, 4); // Usually dim 1 for IDs
            if (n_medoids > 0) {
                uint32_t first_medoid;
                m_in.read((char*)&first_medoid, 4);
                start_node = first_medoid;
                std::cout << "Auto-Detected Start Node from " << medoids_file << ": " << start_node << std::endl;
            }
        } else {
            // Read Header from Graph File
            // Header: [N(8), Dim(8), Medoid(8), MaxLen(8)] or [Magic(8), N(8)...]
            std::string header_path = graph_path;
            std::ifstream test_open(header_path);
            if (!test_open.good()) {
                // If direct open fails, try appending suffix
                std::string alt = graph_path + "_disk.index";
                std::ifstream test_alt(alt);
                if (test_alt.good()) {
                    header_path = alt;
                }
            }
            test_open.close();

            diskann::DiskIndexMetadata disk_metadata;
            std::string metadata_error;
            if (diskann::load_disk_index_metadata(header_path, disk_metadata, metadata_error)) {
                start_node = static_cast<uint32_t>(disk_metadata.medoid);
                std::cout << "Auto-Detected Start Node from Header: " << start_node << std::endl;
            } else {
                 std::cerr << "[Warning] Failed to read disk index metadata from " << header_path
                           << ": " << metadata_error << std::endl;
            }
        }
    }

    // Fallback
    if (start_node == 0) {
        std::cout << "Warning: Could not detect start node or it is 0. Using 0." << std::endl;
    }

    // ------------------------------------------------------------------
    // 1. Initialize Graph Source
    // ------------------------------------------------------------------
    // Current reruns use the COMBINED DiskANN-compatible layout only.
    // The fixed/separate path is retained as an experimental branch and is not
    // considered part of the validated rerun pipeline.

    std::unique_ptr<diskann::GraphSource> graph_source;
    std::shared_ptr<AlignedFileReader> reader = std::make_shared<LinuxAlignedFileReader>();

    uint64_t num_nodes = 0;

    if (use_fixed_layout) {
        std::cout << "Using Fixed-R Layout. R=" << R << std::endl;
        std::string graph_file = graph_path.find("_graph.bin") == std::string::npos ? graph_path + "_graph.bin" : graph_path;

        // Get num nodes from file size
        size_t file_sz = get_file_size(graph_file);
        size_t node_size = 4 + 4 + R * 4; // Header + Count + Neighbors
        num_nodes = (file_sz - 16) / node_size; // Assuming 16 byte header for graph file

        auto fs = std::make_unique<diskann::FixedGraphSource>(R);
        fs->load(graph_file, reader, num_nodes);
        graph_source = std::move(fs);
        reader->open(graph_file);

    } else {
        std::cout << "Using DiskANN Combined Layout." << std::endl;
        std::string index_file = graph_path.find("_disk.index") == std::string::npos ? graph_path + "_disk.index" : graph_path;

        diskann::DiskIndexMetadata disk_metadata;
        std::string metadata_error;
        if (!diskann::load_disk_index_metadata(index_file, disk_metadata, metadata_error)) {
            std::cerr << "Failed to read disk index metadata from " << index_file << ": " << metadata_error
                      << std::endl;
            return 1;
        }

        num_nodes = disk_metadata.num_points;
        const uint64_t dims = disk_metadata.data_dim;
        const uint64_t max_node_len = disk_metadata.max_node_len;
        const uint64_t disk_bytes_per_point = dims * sizeof(float);

        uint32_t max_degree = (uint32_t)((max_node_len - disk_bytes_per_point - 4) / 4);

        auto ds = std::make_unique<diskann::DiskANNGraphSource>(max_node_len, max_degree, dims,
                                                                disk_bytes_per_point);
        ds->load(index_file, reader, num_nodes);
        graph_source = std::move(ds);
        reader->open(index_file);
    }

    std::cout << "Number of Nodes: " << num_nodes << std::endl;

    // 2. BFS Traversal (Priority 1)
    std::cout << "Step 1: Running BFS (" << bfs_hops << " hops)..." << std::endl;
    tsl::robin_set<uint32_t> bfs_selected;
    std::vector<uint32_t> bfs_frontier;

    // Start BFS
    std::cout << "Starting BFS from node " << start_node << std::endl;
    bfs_selected.insert(start_node);
    bfs_frontier.push_back(start_node);

    IOContext ctx;
    reader->ensure_thread_registered();
    ctx = reader->get_ctx();

    uint32_t limit_hops = (mode == "bfs") ? 999999 : bfs_hops;
    size_t bfs_batch_size = 512;
    size_t node_buf_sz = graph_source->get_read_buffer_size();
    size_t per_node_aligned = (node_buf_sz + 4095) / 4096 * 4096;
    char* batch_buf = (char*)std::aligned_alloc(4096, bfs_batch_size * per_node_aligned);

    for(uint32_t h=0; h<limit_hops; ++h) {
        if(bfs_frontier.empty()) break;
        std::vector<uint32_t> next_frontier;
        std::cout << "  Hop " << h+1 << " Processing " << bfs_frontier.size() << " nodes." << std::endl;

        for(size_t i=0; i<bfs_frontier.size(); i+=bfs_batch_size) {
             size_t count = std::min<size_t>(bfs_batch_size, bfs_frontier.size() - i);
             std::vector<AlignedRead> reqs;
             reqs.reserve(count);

             for(size_t k=0; k<count; ++k) {
                 uint32_t u = bfs_frontier[i+k];
                 AlignedRead req;
                 graph_source->prepare_node_read(u, req, batch_buf + k*per_node_aligned);
                 reqs.push_back(req);
             }

             reader->read(reqs, ctx);

             // Process
             for(size_t k=0; k<count; ++k) {
                 uint32_t u = bfs_frontier[i+k];
                 uint32_t neighbor_count = 0;
                 const uint32_t* nbrs =
                     graph_source->get_neighbors_ptr(u, batch_buf + k * per_node_aligned, neighbor_count);

                 for(uint32_t idx = 0; idx < neighbor_count; ++idx) {
                    uint32_t v = nbrs[idx];
                    if(bfs_selected.find(v) == bfs_selected.end()) {
                        bfs_selected.insert(v);
                        next_frontier.push_back(v);
                        if(bfs_selected.size() >= num_nodes_to_cache) {
                             std::free(batch_buf);
                             goto bfs_done;
                        }
                    }
                }
             }
        }
        std::cout << "  BFS Frontier Size: " << next_frontier.size() << " (Accumulated: " << bfs_selected.size() << ")" << std::endl;
        bfs_frontier = next_frontier;
        if(bfs_selected.size() >= num_nodes_to_cache) break;
    }
    std::free(batch_buf);
    bfs_done:

    std::cout << "BFS Selected Count: " << bfs_selected.size() << std::endl;

    // If BFS filled everything OR mode is 'bfs', save and exit
    if(mode == "bfs" || bfs_selected.size() >= num_nodes_to_cache) {
        std::cout << "Saving cache list..." << std::endl;
        std::ofstream out(output_file);
        for(uint32_t u : bfs_selected) out << u << "\n";
        out.close();
        return 0;
    }

    // 3. In-Degree Calculation (Priority 2)
    std::cout << "Step 2: Calculating Global In-Degrees via Streaming Scan..." << std::endl;

    // In-degree counter.
    std::vector<uint32_t> in_degrees;
    try {
        in_degrees.resize(num_nodes, 0);
    } catch (std::bad_alloc& e) {
        std::cerr << "Failed to allocate in-degree counters. Not enough RAM." << std::endl;
        return 1;
    }

    // Get Batch Size
    uint32_t io_batch_size = 256;
    if(const char* env_bs = std::getenv("SIGMA_IO_BATCH_SIZE")) {
        io_batch_size = (uint32_t)atoi(env_bs);
        if(io_batch_size == 0) io_batch_size = 256;
    }
    std::cout << "IO Batch Size: " << io_batch_size << std::endl;
    const uint32_t max_degree = graph_source->get_max_degree();

    auto t_start = std::chrono::high_resolution_clock::now();
    std::atomic<bool> scan_failed(false);
    std::mutex scan_error_mu;
    std::string scan_error;

    #pragma omp parallel
    {
        try
        {
            reader->ensure_thread_registered();
            IOContext t_ctx = reader->get_ctx();

            size_t node_buf_size = graph_source->get_read_buffer_size();

            // Batch Buffer: batch_size * element_size
            // aligned_alloc requires size multiple of alignment
            size_t per_node_alloc = (node_buf_size + 4095) / 4096 * 4096;
            size_t total_alloc = io_batch_size * per_node_alloc;

            char* t_buf = (char*)std::aligned_alloc(4096, total_alloc);

            tsl::robin_map<uint32_t, uint32_t> local_increments;
            const size_t local_bucket_hint =
                static_cast<size_t>(io_batch_size) *
                static_cast<size_t>(std::min<uint32_t>(max_degree > 0 ? max_degree : 32, 64));
            local_increments.reserve(std::max<size_t>(local_bucket_hint, 1024));

            #pragma omp for schedule(dynamic, 1)
            for(uint64_t i=0; i<num_nodes; i+=io_batch_size) {
                if (scan_failed.load(std::memory_order_relaxed)) {
                    continue;
                }

                uint64_t count = std::min<uint64_t>(io_batch_size, num_nodes - i);

                std::vector<AlignedRead> reqs;
                reqs.reserve(count);

                for(uint64_t k=0; k<count; ++k) {
                    AlignedRead req;
                    graph_source->prepare_node_read((uint32_t)(i+k), req, t_buf + k * per_node_alloc);
                    reqs.push_back(req);
                }

                reader->read(reqs, t_ctx);
                local_increments.clear();

                static std::atomic<uint64_t> progress_counter(0);

                for(uint64_t k=0; k<count; ++k) {
                    uint32_t neighbor_count = 0;
                    const uint32_t* nbrs =
                        graph_source->get_neighbors_ptr((uint32_t)(i + k), t_buf + k * per_node_alloc, neighbor_count);
                    for(uint32_t idx = 0; idx < neighbor_count; ++idx) {
                        uint32_t neighbors_id = nbrs[idx];
                        if (neighbors_id < num_nodes) {
                            auto it = local_increments.find(neighbors_id);
                            if (it == local_increments.end()) {
                                local_increments.emplace(neighbors_id, 1);
                            } else {
                                ++(it.value());
                            }
                        }
                    }
                }

                for (const auto& entry : local_increments) {
                    #pragma omp atomic
                    in_degrees[entry.first] += entry.second;
                }

                uint64_t p = progress_counter.fetch_add(count) + count;
                if (p % 100000 < io_batch_size * 64) {
                    if (p % 1000000 < count * 2) { // Every ~1M
                        std::cout << "  Scanned " << p << " / " << num_nodes << " nodes..." << std::endl;
                    }
                }
            }

            std::free(t_buf);
        }
        catch (const std::exception& e)
        {
            if (!scan_failed.exchange(true))
            {
                std::lock_guard<std::mutex> lk(scan_error_mu);
                scan_error = e.what();
            }
        }
        catch (...)
        {
            if (!scan_failed.exchange(true))
            {
                std::lock_guard<std::mutex> lk(scan_error_mu);
                scan_error = "unknown exception in cache generator worker";
            }
        }
    }

    if (scan_failed.load())
    {
        std::cerr << "In-degree calculation aborted: " << scan_error << std::endl;
        return 1;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "In-degree calculation took: " << std::chrono::duration<double>(t_end-t_start).count() << "s" << std::endl;

    // 4. Sorting and Filling (Memory Optimized)
    std::cout << "Step 3: Selecting Top In-Degree Nodes (Memory Optimized)..." << std::endl;

    // We only need 'needed' top nodes. Use a min-heap of size 'needed'.
    // Memory usage: sizeof(NodeIndegree) * needed (negligible compared to full vector)

    uint32_t needed = (bfs_selected.size() >= num_nodes_to_cache) ? 0 : num_nodes_to_cache - (uint32_t)bfs_selected.size();

    std::vector<uint32_t> final_cache;
    final_cache.reserve(num_nodes_to_cache);
    // Add BFS nodes first
    for(auto x : bfs_selected) final_cache.push_back(x);

    if (needed > 0) {
        // Min-heap to keep largest 'needed' elements
        std::priority_queue<NodeIndegree, std::vector<NodeIndegree>, std::greater<NodeIndegree>> pq;

        for(uint64_t i=0; i<num_nodes; ++i) {
            if(bfs_selected.find(i) == bfs_selected.end()) {
                uint32_t deg = in_degrees[i];
                if (pq.size() < needed) {
                    pq.push({(uint32_t)i, deg});
                } else if (deg > pq.top().degree) {
                    pq.pop();
                    pq.push({(uint32_t)i, deg});
                }
            }
        }

        // Extract from heap
        std::vector<uint32_t> top_k;
        top_k.reserve(needed);
        while(!pq.empty()) {
            top_k.push_back(pq.top().id);
            pq.pop();
        }
        // Heap gives smallest first (reverse order of top K).
        // We usually want highest priority first, but for cache list order might not matter
        // except for maybe linearity? Let's reverse to be safe (highest degree first).
        std::reverse(top_k.begin(), top_k.end());

        final_cache.insert(final_cache.end(), top_k.begin(), top_k.end());
    }

    // 5. Save
    std::cout << "Saving " << final_cache.size() << " nodes to " << output_file << std::endl;
    std::ofstream out(output_file);
    for(uint32_t u : final_cache) out << u << "\n";
    out.close();

    return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
}

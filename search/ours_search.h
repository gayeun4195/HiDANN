#pragma once
#include "common_includes.h"
#include "aligned_file_reader.h"
#include "pq.h"
#include "distance.h"
#include "neighbor.h"
#include "percentile_stats.h"
#include "scratch.h"
#include "timer.h"
#include "tsl/robin_map.h"
#include "ours_data_interfaces.h" // Include abstractions

namespace diskann {

template <typename T, typename LabelT = uint32_t>
class SigmaSearch {
public:
    SigmaSearch(std::shared_ptr<AlignedFileReader> reader, Metric metric);
    ~SigmaSearch();

    // Load method
    // Now accepts layout parameters or detects them?
    // For simplicity, we just pass basic index prefix.
    void load(const std::string& index_prefix, uint32_t num_threads);

    // Main search function
    void cached_beam_search_explicit(const T* query, uint64_t k_search, uint64_t l_search,
                                     uint64_t* indices, float* distances,
                                     uint64_t beam_width, bool use_filter, LabelT filter_label,
                                     uint32_t io_limit, bool refine_total,
                                     uint32_t early_exit_streak, uint32_t fixed_refine_count,
                                     double fixed_refine_ratio, QueryStats* stats);

    // Overloads
    void cached_beam_search_explicit(const T* query, uint64_t k_search, uint64_t l_search,
                                     uint64_t* indices, float* distances,
                                     uint64_t beam_width, uint32_t io_limit, bool refine_total,
                                     uint32_t early_exit_streak, uint32_t fixed_refine_count,
                                     double fixed_refine_ratio, QueryStats* stats);

    // Cache helper
    void load_cache_list(std::vector<uint32_t>& node_list);
    void cache_bfs_levels(uint64_t num_nodes, std::vector<uint32_t>& node_list);

    uint64_t num_medoids() const {
        return _num_medoids;
    }

    // Manual Medoid Override
    void set_start_node(uint32_t node_id) {
        if (_medoids.size() == 0) _medoids.resize(1);
        _medoids[0] = node_id;
    }

private:
   // Members
   std::shared_ptr<AlignedFileReader> _reader;
   std::unique_ptr<GraphSource> _graph_source;
   std::unique_ptr<VectorSource> _vector_source;

   Metric _metric;
   FixedChunkPQTable _pq_table;

   // Graph/Index basic info
   uint64_t _num_points = 0;
   uint64_t _max_node_len = 0;
   uint32_t _nnodes_per_sector = 0;
   uint32_t _max_degree = 0;

   uint64_t _data_dim = 0;
   uint64_t _aligned_dim = 0;
   uint64_t _num_chunks = 0;

   // Compressed data
   std::unique_ptr<uint8_t[]> _pq_data;

   // Medoids
   uint64_t _num_medoids = 0;
   std::vector<uint32_t> _medoids;
   std::unique_ptr<float[]> _centroid_data;

   // Cache definitions
   // NL-only cache: node id -> offset into _cache_pool.
    std::vector<uint32_t> _cache_pool;         // Unified flat buffer [Count, N1, N2...]
    std::vector<uint32_t> _cache_hash_keys;    // Compact partial-cache lookup keys
    std::vector<uint32_t> _cache_hash_offsets; // Compact partial-cache lookup offsets
    std::vector<uint32_t> _cache_offsets;      // Dense lookup offsets for full/dense cache
    bool _use_offset_cache_mode = false;
    static constexpr uint32_t CACHE_EMPTY = 0xFFFFFFFFu;

   // Helpers
   std::shared_ptr<Distance<T>> _dist_cmp;
   std::shared_ptr<Distance<float>> _dist_cmp_float;

   ConcurrentQueue<SSDThreadData<T>*> _thread_data;

   // Helper methods
   // Removed old offsets helpers as they are in Source classes now
   static uint64_t compact_cache_hash(uint32_t id);
   void init_compact_cache_hash(size_t num_entries);
   void insert_cache_lookup(uint32_t id, uint32_t pool_offset);
   bool find_cache_lookup(uint32_t id, uint32_t &pool_offset) const;
};

}

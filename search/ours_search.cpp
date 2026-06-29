#include <memory>
#include "ours_search.h"
#include "utils.h"
#include "pq_scratch.h"
#include <fcntl.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace diskann {

namespace {
void drop_file_cache_best_effort(const std::string &path) {
#ifndef _WINDOWS
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return;
    }
    int ret = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (ret != 0) {
        diskann::cerr << "WARNING: posix_fadvise(DONTNEED) failed for " << path << ": " << std::strerror(ret)
                      << std::endl;
    }
    ::close(fd);
#else
    (void) path;
#endif
}
}

template <typename T, typename LabelT>
SigmaSearch<T, LabelT>::SigmaSearch(std::shared_ptr<AlignedFileReader> reader, Metric metric)
    : _reader(reader), _metric(metric) {
    _dist_cmp.reset(get_distance_function<T>(metric));
    _dist_cmp_float.reset(get_distance_function<float>(metric));
}

template <typename T, typename LabelT>
SigmaSearch<T, LabelT>::~SigmaSearch() {
    SSDThreadData<T> *ptr;
    while ((ptr = _thread_data.pop()) != nullptr) {
        delete ptr;
    }
}

template <typename T, typename LabelT>
uint64_t SigmaSearch<T, LabelT>::compact_cache_hash(uint32_t id) {
    uint64_t x = static_cast<uint64_t>(id);
    x ^= x >> 16;
    x *= 0x9E3779B97F4A7C15ull;
    x ^= x >> 32;
    return x;
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::init_compact_cache_hash(size_t num_entries) {
    if (num_entries == 0) {
        return;
    }

    // Keep load factor <= 0.75. With uint32 key+offset arrays, this stays under
    // the 16 bytes/cache-node lookup allowance used by the experiment scripts.
    const size_t capacity = (num_entries * 4 + 2) / 3 + 1;
    _cache_hash_keys.assign(capacity, CACHE_EMPTY);
    _cache_hash_offsets.assign(capacity, CACHE_EMPTY);
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::insert_cache_lookup(uint32_t id, uint32_t pool_offset) {
    if (_cache_hash_keys.empty()) {
        throw ANNException("compact cache hash table is not initialized", -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (id == CACHE_EMPTY) {
        throw ANNException("node id collides with compact cache sentinel", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const uint64_t cap = _cache_hash_keys.size();
    uint64_t pos = compact_cache_hash(id) % cap;
    for (uint64_t probes = 0; probes < cap; ++probes) {
        uint32_t key = _cache_hash_keys[pos];
        if (key == CACHE_EMPTY || key == id) {
            _cache_hash_keys[pos] = id;
            _cache_hash_offsets[pos] = pool_offset;
            return;
        }
        pos++;
        if (pos == cap) {
            pos = 0;
        }
    }

    throw ANNException("compact cache hash table is full", -1, __FUNCSIG__, __FILE__, __LINE__);
}

template <typename T, typename LabelT>
bool SigmaSearch<T, LabelT>::find_cache_lookup(uint32_t id, uint32_t &pool_offset) const {
    if (_cache_hash_keys.empty()) {
        return false;
    }

    const uint64_t cap = _cache_hash_keys.size();
    uint64_t pos = compact_cache_hash(id) % cap;
    for (uint64_t probes = 0; probes < cap; ++probes) {
        uint32_t key = _cache_hash_keys[pos];
        if (key == CACHE_EMPTY) {
            return false;
        }
        if (key == id) {
            pool_offset = _cache_hash_offsets[pos];
            return pool_offset != CACHE_EMPTY;
        }
        pos++;
        if (pos == cap) {
            pos = 0;
        }
    }

    return false;
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::load(const std::string& index_prefix, uint32_t num_threads) {
    // Current rerun scripts exercise the COMBINED DiskANN-compatible layout only.
    // The SEPARATE branch is kept for future work, but it is not part of the active
    // experiment path and should be treated as experimental until revalidated.

    std::string disk_index_file = index_prefix + "_disk.index";
    std::string pq_compressed_file = index_prefix + "_pq_compressed.bin";
    std::string pq_table_file = index_prefix + "_pq_pivots.bin";

    DiskIndexMetadata disk_metadata;
    std::string metadata_error;
    if (!load_disk_index_metadata(disk_index_file, disk_metadata, metadata_error)) {
        throw ANNException("Invalid disk index metadata for " + disk_index_file + ": " + metadata_error,
                           -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    _num_points = disk_metadata.num_points;
    _data_dim = disk_metadata.data_dim;
    const uint32_t header_medoid = static_cast<uint32_t>(disk_metadata.medoid);
    _max_node_len = disk_metadata.max_node_len;
    if (_max_node_len <= 4096) {
        _nnodes_per_sector = 4096 / _max_node_len;
    } else {
        _nnodes_per_sector = 0;
    }

    _aligned_dim = ROUND_UP(_data_dim, 8);

    // Load PQ
    _pq_table.load_pq_centroid_bin(pq_table_file.c_str(), 0);
    _num_chunks = _pq_table.get_num_chunks();

    size_t pq_num, pq_dim;
    load_bin<uint8_t>(pq_compressed_file, _pq_data, pq_num, pq_dim);
    drop_file_cache_best_effort(pq_compressed_file);
    if(pq_num != _num_points) {
         // mismtach?
    }

    // Initialize Sources
    const char* layout = std::getenv("SIGMA_LAYOUT");
    if (layout && std::string(layout) == "SEPARATE") {
         // Experimental path: not used in current reruns.
         uint32_t R = 70; // Default
         const char* env_R = std::getenv("SIGMA_R");
         if(env_R) R = atoi(env_R);
         _max_degree = R;

         _graph_source = std::make_unique<FixedGraphSource>(R);
         _vector_source = std::make_unique<BinVectorSource>();

         // Filenames for separate layout?
         // prefix_graph.bin, prefix_vectors.bin
         _graph_source->load(index_prefix + "_graph.bin", _reader, _num_points);
         _vector_source->load(index_prefix + "_vectors.bin", _reader, _data_dim, _num_points);
    } else {
         // Standard DiskANN
         // We need max degree?
         // DiskANN disk layout stores raw vector bytes; aligned dim is used only
         // after vectors are copied into in-memory scratch/cache buffers.
         uint64_t disk_bytes_per_point = _data_dim * sizeof(T);
         uint32_t max_degree = (uint32_t)((_max_node_len - disk_bytes_per_point - 4) / 4);
         _max_degree = max_degree;

         _graph_source = std::make_unique<DiskANNGraphSource>(_max_node_len, max_degree, _data_dim,
                                                              disk_bytes_per_point);
         _vector_source = std::make_unique<DiskANNVectorSource>(_max_node_len, _data_dim);

         // CRITICAL FIX: Open the reader!
         _reader->open(disk_index_file);
         // _reader->register_thread(); // Test registration state

         _graph_source->load(disk_index_file, _reader, _num_points);
         _vector_source->load(disk_index_file, _reader, _data_dim, _num_points);
    }

    // Medoids
    std::string medoids_file = index_prefix + "_medoids.bin";
    std::string centroids_file = index_prefix + "_centroids.bin";

    if (file_exists(medoids_file)) {
        size_t nm, nd;
        uint32_t* m_ptr = nullptr;
        diskann::load_bin<uint32_t>(medoids_file, m_ptr, nm, nd);
        _medoids.assign(m_ptr, m_ptr + nm * nd);
        _num_medoids = nm * nd;
        delete[] m_ptr;

        _centroid_data.reset(new float[_num_medoids * _aligned_dim]);
        std::memset(_centroid_data.get(), 0, _num_medoids * _aligned_dim * sizeof(float));

        if (file_exists(centroids_file)) {
            size_t c_num, c_dim, c_aligned_dim;
            float* c_ptr = nullptr;
            diskann::load_aligned_bin<float>(centroids_file, c_ptr, c_num, c_dim, c_aligned_dim);
            std::memcpy(_centroid_data.get(), c_ptr, c_num * c_aligned_dim * sizeof(float));
            diskann::aligned_free(c_ptr);
        } else {
             // Read from Vector Source
             std::vector<char> vec_buf(_vector_source->get_read_buffer_size());

             // Ensure this thread is registered for IO
             _reader->ensure_thread_registered();
             IOContext ctx = _reader->get_ctx();

             for(size_t i=0; i<_num_medoids; ++i) {
                  AlignedRead req;
                  std::vector<AlignedRead> reqs;
                  _vector_source->prepare_batch_read({_medoids[i]}, {vec_buf.data()}, reqs);
                  _reader->read(reqs, ctx);

                  const char* v_ptr = _vector_source->get_vector_ptr(_medoids[i], vec_buf.data());
                  for(size_t d=0; d<_data_dim; ++d) {
                       _centroid_data[i*_aligned_dim + d] = (float)((const T*)v_ptr)[d];
                  }
             }
        }
    } else {
        _medoids.resize(1);
        _medoids[0] = header_medoid;
        _num_medoids = 1;

        // Load single medoid centroid
        _centroid_data.reset(new float[_aligned_dim]);
        std::memset(_centroid_data.get(), 0, _aligned_dim * sizeof(float));

         // Read from Vector Source
         std::vector<char> vec_buf(_vector_source->get_read_buffer_size());

         // Ensure this thread is registered for IO
         _reader->ensure_thread_registered();
         IOContext ctx = _reader->get_ctx();

         AlignedRead req;
         std::vector<AlignedRead> reqs;
         _vector_source->prepare_batch_read({_medoids[0]}, {vec_buf.data()}, reqs);
         _reader->read(reqs, ctx);

         const char* v_ptr = _vector_source->get_vector_ptr(_medoids[0], vec_buf.data());
         for(size_t d=0; d<_data_dim; ++d) {
              _centroid_data[d] = (float)((const T*)v_ptr)[d];
         }
    }

    // Thread Data
    if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
    SSDThreadData<T> *ptr;
    while ((ptr = _thread_data.pop()) != nullptr) delete ptr;

    for (uint32_t i = 0; i < num_threads; ++i) {
        SSDThreadData<T>* data = new SSDThreadData<T>(_aligned_dim, 2048);
        _thread_data.push(data);
    }
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::cached_beam_search_explicit(const T* query1, uint64_t k_search,
                                                         uint64_t l_search, uint64_t* indices,
                                                         float* distances, uint64_t beam_width,
                                                         uint32_t io_limit, bool refine_total,
                                                         uint32_t early_exit_streak,
                                                         uint32_t fixed_refine_count,
                                                         double fixed_refine_ratio, QueryStats* stats) {
    cached_beam_search_explicit(query1, k_search, l_search, indices, distances, beam_width,
                                false, (LabelT)0, io_limit, refine_total, early_exit_streak,
                                fixed_refine_count, fixed_refine_ratio, stats);
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::cached_beam_search_explicit(const T* query1, uint64_t k_search,
                                                         uint64_t l_search, uint64_t* indices,
                                                         float* distances, uint64_t beam_width,
                                                         bool use_filter, LabelT filter_label,
                                                         uint32_t io_limit, bool refine_total,
                                                         uint32_t early_exit_streak,
                                                         uint32_t fixed_refine_count,
                                                         double fixed_refine_ratio, QueryStats* stats) {

    ScratchStoreManager<SSDThreadData<T>> manager(this->_thread_data);
    auto data = manager.scratch_space();

    // Initialize context from reader (Critical fix for Linux AIO)
    _reader->ensure_thread_registered();
    data->ctx = _reader->get_ctx();
    IOContext &ctx = data->ctx;

    auto *query_scratch = &data->scratch;
    auto *pq_query_scratch = query_scratch->pq_scratch();

    query_scratch->reset();

    // 1. Prepare Query (Normalize etc)
    float query_norm = 0.0f;
    T *aligned_query_T = query_scratch->aligned_query_T();
    float *query_float = pq_query_scratch->aligned_query_float;
    float *query_rotated = pq_query_scratch->rotated_query;

    if (_metric == diskann::Metric::INNER_PRODUCT || _metric == diskann::Metric::COSINE) {
        // ... (Normalization logic same as before)
        const uint64_t inherent_dim = (_metric == diskann::Metric::COSINE) ? _data_dim : (uint64_t)(_data_dim - 1);
        for (size_t i = 0; i < inherent_dim; ++i) {
            aligned_query_T[i] = query1[i];
            query_norm += (float)query1[i] * (float)query1[i];
        }
        if (_metric == diskann::Metric::INNER_PRODUCT) aligned_query_T[_data_dim - 1] = 0;
        query_norm = std::sqrt(query_norm);
        for (size_t i = 0; i < inherent_dim; ++i) {
             aligned_query_T[i] = (T)((float)aligned_query_T[i] / (query_norm > 0 ? query_norm : 1.0f));
        }
        pq_query_scratch->initialize(_data_dim, aligned_query_T);
    } else {
        for (size_t i = 0; i < _data_dim; ++i) aligned_query_T[i] = query1[i];
        pq_query_scratch->initialize(_data_dim, aligned_query_T);
    }
    if (_data_dim < _aligned_dim) {
        std::memset(aligned_query_T + _data_dim, 0, (_aligned_dim - _data_dim) * sizeof(T));
    }

    _pq_table.preprocess_query(query_rotated);
    float *pq_dists = pq_query_scratch->aligned_pqtable_dist_scratch;
    _pq_table.populate_chunk_distances(query_rotated, pq_dists);

    float *dist_scratch = pq_query_scratch->aligned_dist_scratch;
    uint8_t *pq_coord_scratch = pq_query_scratch->aligned_pq_coord_scratch;

    auto compute_dists = [this, pq_coord_scratch, pq_dists](const uint32_t *ids, const uint64_t n_ids,
                                                            float *dists_out) {
        diskann::aggregate_coords(ids, n_ids, this->_pq_data.get(), this->_num_chunks, pq_coord_scratch);
        diskann::pq_dist_lookup(pq_coord_scratch, n_ids, this->_num_chunks, pq_dists, dists_out);
    };

    Timer query_timer, io_timer, cpu_timer;

    tsl::robin_set<uint64_t> &visited = query_scratch->visited;
    NeighborPriorityQueue &retset = query_scratch->retset;
    retset.reserve(l_search);

    // Init Search
    uint32_t best_medoid = _medoids[0];
    float best_dist = std::numeric_limits<float>::max();

    if (_num_medoids > 1) {
         for(size_t i=0; i<_num_medoids; ++i) {
             float d = _dist_cmp_float->compare(query_float, _centroid_data.get() + i * _aligned_dim, (uint32_t)_aligned_dim);
             if (d < best_dist) {
                 best_dist = d;
                 best_medoid = _medoids[i];
             }
         }
    }

    compute_dists(&best_medoid, 1, dist_scratch);
    retset.insert(Neighbor(best_medoid, dist_scratch[0]));
    visited.insert(best_medoid);

    uint32_t hops = 0, num_ios = 0;

    std::vector<uint32_t> frontier; frontier.reserve(2 * beam_width);
    std::vector<AlignedRead> frontier_read_reqs; frontier_read_reqs.reserve(2 * beam_width);

    // reuse sector scratch for node parsing
    char *sector_scratch = query_scratch->sector_scratch;

    // 2. Beam Search Loop
    Timer traversal_timer;


    while (retset.has_unexpanded_node() && num_ios < io_limit) {
        frontier.clear();
        frontier_read_reqs.clear();
        if (stats) stats->n_beam_rounds++;

        uint32_t num_seen = 0;
        // Collect Frontier
        while (retset.has_unexpanded_node() && frontier.size() < beam_width && num_seen < beam_width) {
            auto nbr = retset.closest_unexpanded();
            num_seen++;

            // CACHE LOOKUP (NL Cache)
            // CACHE LOOKUP (Hybrid CSR)
            bool found_in_cache = false;
            uint32_t pool_offset32 = 0;

            if (_use_offset_cache_mode) {
                if (nbr.id < _cache_offsets.size()) {
                    pool_offset32 = _cache_offsets[nbr.id];
                    found_in_cache = (pool_offset32 != CACHE_EMPTY);
                }
            } else {
                found_in_cache = find_cache_lookup(nbr.id, pool_offset32);
            }
            // if (stats) stats->overhead_us += cpu_timer.elapsed(); // End Overhead

            if (found_in_cache) {
                if (stats) stats->n_cache_hits++;

                // Decode from Flat Buffer: [Count, N1, N2...]
                // cpu_timer.reset();
                uint64_t pool_offset = static_cast<uint64_t>(pool_offset32);
                uint32_t count = _cache_pool[pool_offset];
                const uint32_t* neighbors = &_cache_pool[pool_offset + 1];
                // if (stats) stats->overhead_us += cpu_timer.elapsed();

                // Process Cached Neighbors
                if (count > 0) {
                     // Compute PQ Dists in batch
                     // optimization: Reuse dist_scratch (pre-allocated per thread, size >= MAX_DEGREE)
                     // vector allocation here caused massive malloc contention in multi-threaded env.
                     cpu_timer.reset();
                     compute_dists(neighbors, count, dist_scratch);
                     for(uint32_t i=0; i<count; ++i) {
                         if (visited.insert(neighbors[i]).second) {
                             retset.insert(Neighbor(neighbors[i], dist_scratch[i]));
                         }
                     }
                     if(stats) { stats->cpu_us += cpu_timer.elapsed(); stats->n_cmps += count; }
                }
            } else {
                // cpu_timer.reset();
                frontier.push_back(nbr.id);
                // if (stats) stats->overhead_us += cpu_timer.elapsed();
            }
        }

        // Process Frontier (Disk Read)
        if (!frontier.empty()) {
            if (stats) { stats->n_hops++; stats->n_graph_io_batches++; }

            // Use GraphSource to prepare reads
            // cpu_timer.reset(); // Reuse cpu_timer for overhead
            uint32_t buf_stride = _graph_source->get_read_buffer_size();
            for (size_t i=0; i<frontier.size(); ++i) {
                uint32_t id = frontier[i];
                char* buf = sector_scratch + i * buf_stride;

                // INLINED LOGIC (Optimization 2)
                uint64_t abs_offset;
                if (_nnodes_per_sector > 0) {
                    uint64_t sector_idx = (uint64_t)id / _nnodes_per_sector;
                    uint64_t idx_in_sector = (uint64_t)id % _nnodes_per_sector;
                    abs_offset = (sector_idx + 1) * 4096 + idx_in_sector * _max_node_len;
                } else {
                    uint64_t aligned_node_len = (_max_node_len + 4095) / 4096 * 4096;
                    abs_offset = 4096 + (uint64_t)id * aligned_node_len;
                }

                uint64_t sector_start = (abs_offset / 4096) * 4096;
                uint64_t sector_end = ((abs_offset + _max_node_len + 4096 - 1) / 4096) * 4096;

                frontier_read_reqs.emplace_back(sector_start, (uint32_t)(sector_end - sector_start), buf);

                /*
                AlignedRead req;
                _graph_source->prepare_node_read(id, req, buf);
                frontier_read_reqs.push_back(req);
                */

                if (stats) { stats->n_4k++; stats->n_ios++; stats->n_graph_ios++; }
                num_ios++;
            }
            // if (stats) stats->overhead_us += cpu_timer.elapsed();

            io_timer.reset();
            _reader->read(frontier_read_reqs, ctx);
            if (stats) stats->io_us += io_timer.elapsed();

            // Parse and Expand
            for (size_t i=0; i<frontier.size(); ++i) {
                uint32_t id = frontier[i];
                char* buf = sector_scratch + i * buf_stride; // Unchanged stride logic

                // cpu_timer.reset(); // Start overhead measurement
                uint32_t nnbrs = 0;
                const uint32_t* node_nbrs = _graph_source->get_neighbors_ptr(id, buf, nnbrs);
                // if (stats) stats->overhead_us += cpu_timer.elapsed(); // End overhead measurement

                if (nnbrs > 0) {
                    // Zero-Copy!
                    cpu_timer.reset();
                    compute_dists(node_nbrs, nnbrs, dist_scratch);
                    for(size_t j=0; j<nnbrs; ++j) {
                        uint32_t nbr_id = node_nbrs[j];
                        if (visited.insert(nbr_id).second) {
                            retset.insert(Neighbor(nbr_id, dist_scratch[j]));
                        }
                    }
                    if(stats) { stats->cpu_us += cpu_timer.elapsed(); stats->n_cmps += nnbrs; }
                }
            }
        } else {
            if (stats) stats->n_beam_no_io_rounds++;
        }
        hops++;
    } // End Beam Search
    if (stats) stats->traversal_us += traversal_timer.elapsed();

    // 3. Batch Refinement
    // Reuse scratch storage to avoid per-query candidate vector allocations.
    std::vector<Neighbor> &candidates = query_scratch->full_retset;
    candidates.clear();
    candidates.reserve(retset.size());
    for(size_t i=0; i<retset.size(); ++i) {
        candidates.push_back(retset[i]);
    }
    // NeighborPriorityQueue keeps its active range sorted by approximate distance.

    uint64_t refine_num = std::min<uint64_t>(l_search, candidates.size());
    if (fixed_refine_count > 0) {
        refine_num = std::min<uint64_t>(refine_num, fixed_refine_count);
    } else if (fixed_refine_ratio > 0.0) {
        const uint64_t ratio_refine =
            static_cast<uint64_t>(std::ceil(static_cast<double>(refine_num) * fixed_refine_ratio));
        refine_num = std::min<uint64_t>(refine_num, std::max<uint64_t>(k_search, ratio_refine));
    }
    size_t M = (size_t)refine_num;

    const char* env_bs = std::getenv("BATCH_REFINE_SIZE");
    size_t B = 50;
    if (env_bs) {
        const int parsed_B = atoi(env_bs);
        if (parsed_B > 0) {
            B = static_cast<size_t>(parsed_B);
        }
    }

    Timer refinement_timer;

    // Batch Buffer
    size_t chunk_size = _vector_source->get_read_buffer_size();
    size_t batch_bytes = B * chunk_size;

    // Adaptive Buffer Reuse
    data->resize_refinement_buffer(batch_bytes, 4096);
    char *batch_buf = data->refinement_buffer;

    std::vector<AlignedRead> refine_reqs;
    refine_reqs.reserve(B);
    std::vector<uint32_t> batch_ids;
    batch_ids.reserve(B);
    std::vector<char*> cur_bufs;
    cur_bufs.reserve(B);
    T* aligned_refine_vector = query_scratch->coord_scratch;
    if (_data_dim < _aligned_dim) {
        std::memset(aligned_refine_vector + _data_dim, 0, (_aligned_dim - _data_dim) * sizeof(T));
    }

    if (early_exit_streak == 0) {
        early_exit_streak = 1;
    }

    uint32_t streak_cnt = 0;

    struct TopKNode { float d; uint32_t id; };
    struct TopKCmp {
        bool operator()(const TopKNode& a, const TopKNode& b) const { return a.d < b.d; }
    };
    std::vector<TopKNode> topk_heap;
    topk_heap.reserve((size_t)k_search);

    auto compute_topk_ids = [&](size_t upto, uint64_t k, std::vector<uint32_t>& ids) {
        ids.clear();
        const size_t K = (size_t)std::min<uint64_t>(k, upto);
        if (K == 0) return;

        topk_heap.clear();
        for (size_t i = 0; i < upto; ++i) {
            TopKNode cur{ candidates[i].distance, candidates[i].id };
            if (topk_heap.size() < K) {
                topk_heap.push_back(cur);
                std::push_heap(topk_heap.begin(), topk_heap.end(), TopKCmp());
            } else if (cur.d < topk_heap.front().d) {
                std::pop_heap(topk_heap.begin(), topk_heap.end(), TopKCmp());
                topk_heap.back() = cur;
                std::push_heap(topk_heap.begin(), topk_heap.end(), TopKCmp());
            }
        }
        ids.reserve(K);
        for (auto &n : topk_heap) ids.push_back(n.id);
        std::sort(ids.begin(), ids.end());
    };

    std::vector<uint32_t> prev_topk_ids;
    std::vector<uint32_t> curr_topk_ids;
    prev_topk_ids.reserve((size_t)k_search);
    curr_topk_ids.reserve((size_t)k_search);
    const bool use_dynamic_early_exit = !refine_total && fixed_refine_count == 0 && fixed_refine_ratio == 0.0;
    if (use_dynamic_early_exit) {
        compute_topk_ids(M, k_search, prev_topk_ids);
    }
    for (size_t start = 0; start < M; start += B) {
        size_t bsz = std::min(B, M - start);
        batch_ids.clear();
        cur_bufs.clear();
        for(size_t j=0; j<bsz; ++j) {
            batch_ids.push_back(candidates[start+j].id);
            cur_bufs.push_back(batch_buf + j * chunk_size);
        }

        _vector_source->prepare_batch_read(batch_ids, cur_bufs, refine_reqs);

        if (stats) { stats->n_ios += bsz; stats->n_io_batches++; stats->n_4k += bsz; }
        io_timer.reset();
        _reader->read(refine_reqs, ctx);
        if (stats) stats->io_us += io_timer.elapsed();

        cpu_timer.reset();
        for(size_t j=0; j<bsz; ++j) {
            const char* vec_ptr = _vector_source->get_vector_ptr(batch_ids[j], cur_bufs[j]);
            std::memcpy(aligned_refine_vector, vec_ptr, _data_dim * sizeof(T));
            float d = _dist_cmp->compare(aligned_query_T, aligned_refine_vector, (uint32_t)_aligned_dim);
            candidates[start+j].distance = d; // update exact
        }
        if (stats) stats->cpu_us += cpu_timer.elapsed();

        // Check Early Exit
        if (use_dynamic_early_exit) {
             compute_topk_ids(M, k_search, curr_topk_ids);
             if (curr_topk_ids == prev_topk_ids) {
                 streak_cnt++;
                 if (streak_cnt >= early_exit_streak) {
                     break;
                 }
             } else {
                 streak_cnt = 0;
             }
             prev_topk_ids.swap(curr_topk_ids);
        }
    }

    // Final Sort and Output
    std::partial_sort(candidates.begin(), candidates.begin() + std::min<uint64_t>(k_search, M), candidates.begin() + M);
    uint64_t Kout = std::min<uint64_t>(k_search, M);
    for(uint64_t i=0; i<Kout; ++i) {
        indices[i] = candidates[i].id;
        distances[i] = candidates[i].distance;
        if (_metric == diskann::Metric::INNER_PRODUCT) distances[i] = -distances[i];
    }
    if (stats) stats->refinement_us += refinement_timer.elapsed();
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::load_cache_list(std::vector<uint32_t>& node_list) {
    // Clear old data
    _cache_pool.clear();
    _cache_hash_keys.clear();
    _cache_hash_offsets.clear();
    _cache_offsets.clear();
    _use_offset_cache_mode = false;

    if (node_list.empty()) return;

    const uint32_t reserve_per_node = (_max_degree > 0) ? (_max_degree + 1) : 32;
    const uint64_t expected_pool_entries = static_cast<uint64_t>(node_list.size()) * reserve_per_node;
    if (expected_pool_entries >= CACHE_EMPTY) {
        throw ANNException("cache pool offset does not fit in uint32_t", -1, __FUNCSIG__, __FILE__, __LINE__);
    }

    const char *lookup_mode_env = std::getenv("SIGMA_CACHE_LOOKUP_MODE");
    const std::string lookup_mode = lookup_mode_env ? std::string(lookup_mode_env) : "auto";
    const bool force_offset = (lookup_mode == "offset-vector" || lookup_mode == "offset");
    const bool force_hash = (lookup_mode == "compact-hash" || lookup_mode == "hash" || lookup_mode == "hash-map");

    if (force_offset) {
        _use_offset_cache_mode = true;
    } else if (force_hash) {
        _use_offset_cache_mode = false;
    } else {
        const uint64_t offset_lookup_bytes = _num_points * sizeof(uint32_t);
        const uint64_t compact_hash_bytes = ((static_cast<uint64_t>(node_list.size()) * 4 + 2) / 3 + 1) *
                                            2 * sizeof(uint32_t);
        _use_offset_cache_mode = (node_list.size() == _num_points || offset_lookup_bytes <= compact_hash_bytes);
    }

    _cache_pool.reserve(static_cast<size_t>(expected_pool_entries));
    if (_use_offset_cache_mode) {
        _cache_offsets.assign(static_cast<size_t>(_num_points), CACHE_EMPTY);
        diskann::cout << "HiDANN cache lookup mode: offset-vector, nodes=" << node_list.size()
                      << ", offset_bytes=" << (_num_points * sizeof(uint32_t)) << std::endl;
    } else {
        init_compact_cache_hash(node_list.size());
        diskann::cout << "HiDANN cache lookup mode: compact-hash, nodes=" << node_list.size()
                      << ", hash_bytes=" << ((_cache_hash_keys.size() + _cache_hash_offsets.size()) * sizeof(uint32_t))
                      << std::endl;
    }

    _reader->ensure_thread_registered();
    IOContext &ctx = _reader->get_ctx();

    // Sort in-place for sequential I/O. The caller clears node_list after loading.
    std::sort(node_list.begin(), node_list.end());

    // Optimization: Batch I/O
    // Use env var or default 256
    uint32_t batch_size = 256;
    if(const char* env_bs = std::getenv("SIGMA_IO_BATCH_SIZE")) {
        batch_size = (uint32_t)atoi(env_bs);
        if(batch_size == 0) batch_size = 256;
    }

    uint32_t node_buf_size = _graph_source->get_read_buffer_size();
    // Align the buffer size to 4096 for O_DIRECT requirements if needed,
    // though aligned_alloc handles base pointer, size should be safe.
    // However, to be safe with batch buffer indexing:
    size_t per_node_alloc = (node_buf_size + 4095) / 4096 * 4096;
    size_t total_alloc = batch_size * per_node_alloc;

    char* batch_buf = nullptr;
    alloc_aligned(((void**)&batch_buf), total_alloc, 4096);

    for(size_t i = 0; i < node_list.size(); i += batch_size) {
        size_t current_count = std::min<size_t>(batch_size, node_list.size() - i);

        std::vector<AlignedRead> reqs;
        reqs.reserve(current_count);

        // 1. Prepare Batch Requests
        for(size_t k = 0; k < current_count; ++k) {
            uint32_t id = node_list[i + k];
            AlignedRead req;
            _graph_source->prepare_node_read(id, req, batch_buf + k * per_node_alloc);
            reqs.push_back(req);
        }

        // 2. Submit Batch I/O
        _reader->read(reqs, ctx);

        // 3. Process Batch Parsed Data
        for(size_t k = 0; k < current_count; ++k) {
            uint32_t id = node_list[i + k];
            uint32_t neighbor_count = 0;
            const uint32_t* neighbors =
                _graph_source->get_neighbors_ptr(id, batch_buf + k * per_node_alloc, neighbor_count);

            // Store in Flat Buffer
            uint64_t current_offset = _cache_pool.size();
            if (current_offset >= CACHE_EMPTY) {
                aligned_free(batch_buf);
                throw ANNException("cache pool offset does not fit in uint32_t", -1, __FUNCSIG__, __FILE__,
                                   __LINE__);
            }

            // 3.1 Count
            _cache_pool.push_back(neighbor_count);

            // 3.2 Neighbors
            _cache_pool.insert(_cache_pool.end(), neighbors, neighbors + neighbor_count);

            // 3.3 Index
            if (_use_offset_cache_mode) {
                _cache_offsets[id] = static_cast<uint32_t>(current_offset);
            } else {
                insert_cache_lookup(id, static_cast<uint32_t>(current_offset));
            }
        }
    }

    aligned_free(batch_buf);
}

template <typename T, typename LabelT>
void SigmaSearch<T, LabelT>::cache_bfs_levels(uint64_t num_nodes, std::vector<uint32_t>& node_list) {
    if (num_nodes == 0) return;

    tsl::robin_set<uint32_t> visited;
    std::vector<uint32_t> frontier;

    // Start with medoids
    for(uint32_t m : _medoids) {
        if(visited.find(m) == visited.end()) {
            visited.insert(m);
            frontier.push_back(m);
            node_list.push_back(m);
        }
    }

    std::vector<uint32_t> next_frontier;
    // Use correct context
    _reader->ensure_thread_registered();
    IOContext &ctx = _reader->get_ctx();

    uint32_t batch_size = 256;
    if(const char* env_bs = std::getenv("SIGMA_IO_BATCH_SIZE")) {
        batch_size = (uint32_t)atoi(env_bs);
        if(batch_size == 0) batch_size = 256;
    }

    uint32_t node_buf_size = _graph_source->get_read_buffer_size();
    size_t per_node_alloc = (node_buf_size + 4095) / 4096 * 4096;
    size_t total_alloc = static_cast<size_t>(batch_size) * per_node_alloc;

    char* batch_buf = nullptr;
    alloc_aligned(((void**)&batch_buf), total_alloc, 4096);

    while(node_list.size() < num_nodes && !frontier.empty()) {
        next_frontier.clear();

        // Shuffle frontier to sample randomly? Or strict BFS?
        // Standard BFS is fine.

        for(size_t i = 0; i < frontier.size() && node_list.size() < num_nodes; i += batch_size) {
            size_t current_count = std::min<size_t>(batch_size, frontier.size() - i);
            std::vector<AlignedRead> reqs;
            reqs.reserve(current_count);

            for(size_t k = 0; k < current_count; ++k) {
                uint32_t u = frontier[i + k];
                AlignedRead req;
                _graph_source->prepare_node_read(u, req, batch_buf + k * per_node_alloc);
                reqs.push_back(req);
            }

            _reader->read(reqs, ctx);

            for(size_t k = 0; k < current_count && node_list.size() < num_nodes; ++k) {
                uint32_t u = frontier[i + k];
                uint32_t neighbor_count = 0;
                const uint32_t* nbrs =
                    _graph_source->get_neighbors_ptr(u, batch_buf + k * per_node_alloc, neighbor_count);

                for(uint32_t idx = 0; idx < neighbor_count; ++idx) {
                    uint32_t v = nbrs[idx];
                    if(visited.find(v) == visited.end()) {
                        visited.insert(v);
                        next_frontier.push_back(v);
                        node_list.push_back(v);
                        if(node_list.size() >= num_nodes) break;
                    }
                }
            }
        }
        frontier = next_frontier;
    }
    aligned_free(batch_buf);
}

// Instantiate
template class SigmaSearch<float>;
template class SigmaSearch<int8_t>;
template class SigmaSearch<uint8_t>;

} // namespace diskann

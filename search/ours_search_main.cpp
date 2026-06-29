// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "common_includes.h"
#include <atomic>
#include <fcntl.h>
#include <mutex>
#include <boost/program_options.hpp>

#include "index.h"
#include "disk_utils.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition.h"
#include "ours_search.h"
#include "timer.h"
#include "percentile_stats.h"
#include "program_options_utils.hpp"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#else
#ifdef USE_BING_INFRA
#include "bing_aligned_file_reader.h"
#else
#include "windows_aligned_file_reader.h"
#endif
#endif

#define WARMUP false

namespace po = boost::program_options;

namespace {
void drop_file_cache_best_effort(const std::string &path)
{
#ifndef _WINDOWS
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return;

    int ret = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    if (ret != 0)
    {
        diskann::cerr << "WARNING: posix_fadvise(DONTNEED) failed for " << path << ": " << std::strerror(ret)
                      << std::endl;
    }
    ::close(fd);
#else
    (void) path;
#endif
}
}

void print_stats(std::string category, std::vector<float> percentiles, std::vector<float> results)
{
    diskann::cout << std::setw(20) << category << ": " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(8) << percentiles[s] << "%";
    }
    diskann::cout << std::endl;
    diskann::cout << std::setw(22) << " " << std::flush;
    for (uint32_t s = 0; s < percentiles.size(); s++)
    {
        diskann::cout << std::setw(9) << results[s];
    }
    diskann::cout << std::endl;
}

static inline std::string trim_copy(const std::string &s)
{
    const char *ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos)
        return {};
    const auto e = s.find_last_not_of(ws);
    return s.substr(b, e - b + 1);
}

bool read_node_list_from_txt(const std::string &txt_path, uint64_t num_nodes_to_cache, std::vector<uint32_t> &node_list,
                             std::string *error_message = nullptr)
{
    auto set_error = [&](const std::string &msg) {
        if (error_message != nullptr)
            *error_message = msg;
    };

    node_list.clear();
    if (num_nodes_to_cache == 0)
        return true;

    std::ifstream in(txt_path);
    if (!in)
    {
        set_error("failed to open file");
        return false;
    }

    node_list.reserve(static_cast<size_t>(std::min<uint64_t>(num_nodes_to_cache, std::numeric_limits<size_t>::max())));

    std::string line;
    size_t line_no = 0;
    while (node_list.size() < num_nodes_to_cache && std::getline(in, line))
    {
        ++line_no;
        std::string t = trim_copy(line);
        if (t.empty() || t[0] == '#')
            continue;

        char *endp = nullptr;
        errno = 0;
        unsigned long long v = std::strtoull(t.c_str(), &endp, 10);
        if (errno != 0 || endp == t.c_str() || !trim_copy(std::string(endp)).empty())
        {
            set_error("invalid node id at line " + std::to_string(line_no) + ": " + t);
            return false;
        }
        if (v > std::numeric_limits<uint32_t>::max())
        {
            set_error("node id out of uint32 range at line " + std::to_string(line_no) + ": " + t);
            return false;
        }

        node_list.push_back(static_cast<uint32_t>(v));
    }

    if (node_list.size() != num_nodes_to_cache)
    {
        set_error("expected " + std::to_string(num_nodes_to_cache) + " node ids, found " +
                  std::to_string(node_list.size()));
        return false;
    }

    return true;
}

template <typename T, typename LabelT = uint32_t>
int search_disk_index(diskann::Metric &metric, const std::string &index_path_prefix,
                      const std::string &result_output_prefix, const std::string &query_file, std::string &gt_file,
                      const uint32_t num_threads, const uint32_t recall_at, const uint32_t beamwidth,
                      const uint32_t num_nodes_to_cache, const uint32_t search_io_limit,
                      const std::vector<uint32_t> &Lvec, const float fail_if_recall_below,
                      int64_t start_node_override, bool refine_total, uint32_t early_exit_streak,
                      uint32_t fixed_refine_count, double fixed_refine_ratio, const std::string &query_stats_path)
{
    if (beamwidth == 0)
    {
        diskann::cerr << "Error: beamwidth must be > 0. Automatic beamwidth fallback is disabled for reproducible experiments."
                      << std::endl;
        return -1;
    }
    if (early_exit_streak == 0)
    {
        diskann::cerr << "Error: early_exit_streak must be >= 1." << std::endl;
        return -1;
    }
    if (fixed_refine_ratio < 0.0 || fixed_refine_ratio > 1.0)
    {
        diskann::cerr << "Error: fixed_refine_ratio must be in [0, 1]." << std::endl;
        return -1;
    }
    if (refine_total && (fixed_refine_count > 0 || fixed_refine_ratio > 0.0))
    {
        diskann::cerr << "Error: --refine_total cannot be combined with fixed refinement." << std::endl;
        return -1;
    }
    if (fixed_refine_count > 0 && fixed_refine_ratio > 0.0)
    {
        diskann::cerr << "Error: --fixed_refine_count and --fixed_refine_ratio are mutually exclusive." << std::endl;
        return -1;
    }

    const char* env_bs = std::getenv("BATCH_REFINE_SIZE");
    size_t batch_size = 50;
    if (env_bs)
    {
        const int parsed_batch_size = atoi(env_bs);
        if (parsed_batch_size > 0)
            batch_size = static_cast<size_t>(parsed_batch_size);
    }
    std::string refinement_mode = refine_total ? "full" : "early-exit";
    if (fixed_refine_count > 0)
        refinement_mode = "fixed-count=" + std::to_string(fixed_refine_count);
    else if (fixed_refine_ratio > 0.0)
        refinement_mode = "fixed-ratio=" + std::to_string(fixed_refine_ratio);
    diskann::cout << "Search parameters: #threads: " << num_threads << ", batch size: " << batch_size
                  << ", beamwidth: " << beamwidth
                  << ", refinement: " << refinement_mode
                  << ", early_exit_streak: " << early_exit_streak << std::flush;
    if (search_io_limit == std::numeric_limits<uint32_t>::max())
        diskann::cout << "." << std::endl;
    else
        diskann::cout << ", io_limit: " << search_io_limit << "." << std::endl;

    std::string warmup_query_file = index_path_prefix + "_sample_data.bin";

    // load query bin
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);
    drop_file_cache_best_effort(query_file);

    bool calc_recall_flag = false;
    if (gt_file != std::string("null") && gt_file != std::string("NULL") && file_exists(gt_file))
    {
        diskann::load_truthset(gt_file, gt_ids, gt_dists, gt_num, gt_dim);
        drop_file_cache_best_effort(gt_file);
        if (gt_num != query_num)
        {
            diskann::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
        calc_recall_flag = true;
    }

    std::shared_ptr<AlignedFileReader> reader = nullptr;
#ifdef _WINDOWS
#ifndef USE_BING_INFRA
    reader.reset(new WindowsAlignedFileReader());
#else
    reader.reset(new diskann::BingAlignedFileReader());
    reader->register_thread(); // CRITICAL: Main thread does caching, must register I/O context!
#endif
#else
    reader.reset(new LinuxAlignedFileReader());
#endif

    // USE SIGMA SEARCH
    std::unique_ptr<diskann::SigmaSearch<T, LabelT>> _pFlashIndex(
        new diskann::SigmaSearch<T, LabelT>(reader, metric));

    _pFlashIndex->load(index_path_prefix, num_threads);

    if (start_node_override >= 0) {
        if (_pFlashIndex->num_medoids() > 1) {
            diskann::cerr << "Error: start_node override is disabled when multiple medoids are present."
                          << std::endl;
            return -1;
        }
        diskann::cout << "Overriding start node (medoid) to: " << start_node_override << std::endl;
        _pFlashIndex->set_start_node((uint32_t)start_node_override);
    }

    std::vector<uint32_t> node_list;
    const char *env_cache_txt = std::getenv("CACHE_FILE");
    if (env_cache_txt && *env_cache_txt)
    {
        const std::string cache_txt(env_cache_txt);
        diskann::cout << "Caching " << num_nodes_to_cache << " nodes from file (CACHE_FILE=" << cache_txt << ")\n";
        std::string cache_file_error;
        if (!read_node_list_from_txt(cache_txt, num_nodes_to_cache, node_list, &cache_file_error))
        {
            diskann::cerr << "Invalid CACHE_FILE " << cache_txt << ": " << cache_file_error << std::endl;
            return -1;
        }
        drop_file_cache_best_effort(cache_txt);
    }
    else
    {
        diskann::cout << "Caching " << num_nodes_to_cache << " nodes around medoid(s)" << std::endl;
        _pFlashIndex->cache_bfs_levels(num_nodes_to_cache, node_list);
    }

    _pFlashIndex->load_cache_list(node_list);
    node_list.clear();
    node_list.shrink_to_fit();

    omp_set_num_threads(num_threads);

    diskann::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    diskann::cout.precision(2);

    std::string recall_string = "Recall@" + std::to_string(recall_at);
    diskann::cout << std::setw(6) << "L" << std::setw(12) << "Beamwidth" << std::setw(16) << "QPS" << std::setw(16)
                  << "Mean Latency" << std::setw(16) << "99.9 Latency" << std::setw(16) << "Mean IOs" << std::setw(16)
                  << "Graph IOs" << std::setw(16) << "Cache Hits" << std::setw(16) << "Refined Vecs" << std::setw(16)
                  << "Cache Hit%" << std::setw(16) << "Beam Rounds" << std::setw(16) << "NoIO Beams"
                  << std::setw(16) << "Beam NoIO%" << std::setw(16) << "GraphIO/Beam" << std::setw(16)
                  << "GraphIO/IOBeam" << std::setw(16) << "Mean IO (us)" << std::setw(16) << "CPU (us)" << std::setw(16)
                  << "Traversal (us)" << std::setw(16) << "Refinement (us)" << std::setw(16) << "Batches";
    if (calc_recall_flag)
    {
        diskann::cout << std::setw(16) << recall_string << std::endl;
    }
    else
        diskann::cout << std::endl;
    diskann::cout << "=================================================================="
                     "================================================================="
                  << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());

    const uint32_t effective_beamwidth = beamwidth;
    double best_recall = 0.0;

    std::ofstream query_stats_writer;
    if (!query_stats_path.empty())
    {
        query_stats_writer.open(query_stats_path);
        if (!query_stats_writer)
        {
            diskann::cerr << "Error: failed to open query_stats_path: " << query_stats_path << std::endl;
            diskann::aligned_free(query);
            if (gt_ids != nullptr) delete[] gt_ids;
            if (gt_dists != nullptr) delete[] gt_dists;
            return -1;
        }
        query_stats_writer.setf(std::ios_base::fixed, std::ios_base::floatfield);
        query_stats_writer.precision(6);
        query_stats_writer << "L,query_id,total_us,io_us,cpu_us,traversal_us,refinement_us,n_ios,n_graph_ios,"
                           << "refined_vecs,n_cache_hits,n_beam_rounds,n_beam_no_io_rounds,n_io_batches,"
                           << "n_graph_io_batches,recall_at_" << recall_at << "\n";
    }

    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];

        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);

        auto stats = new diskann::QueryStats[query_num];

        std::vector<uint64_t> query_result_ids_64(recall_at * query_num);
        auto s = std::chrono::high_resolution_clock::now();
        std::atomic<bool> worker_failure(false);
        std::mutex worker_failure_mu;
        std::string worker_failure_msg;

#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            if (worker_failure.load(std::memory_order_relaxed))
            {
                continue;
            }

            try
            {
                reader->ensure_thread_registered();
                auto qs_start = std::chrono::high_resolution_clock::now();
                _pFlashIndex->cached_beam_search_explicit(query + (i * query_aligned_dim), recall_at, L,
                                                            query_result_ids_64.data() + (i * recall_at),
                                                            query_result_dists[test_id].data() + (i * recall_at),
                                                            effective_beamwidth, search_io_limit, refine_total,
                                                            early_exit_streak, fixed_refine_count,
                                                            fixed_refine_ratio, stats + i);
                auto qs_end = std::chrono::high_resolution_clock::now();
                (stats + i)->total_us = std::chrono::duration<double, std::micro>(qs_end - qs_start).count();
            }
            catch (const std::exception &e)
            {
                if (!worker_failure.exchange(true))
                {
                    std::lock_guard<std::mutex> lk(worker_failure_mu);
                    worker_failure_msg = e.what();
                }
            }
            catch (...)
            {
                if (!worker_failure.exchange(true))
                {
                    std::lock_guard<std::mutex> lk(worker_failure_mu);
                    worker_failure_msg = "unknown exception in search worker";
                }
            }
        }

        if (worker_failure.load())
        {
            delete[] stats;
            diskann::cerr << "Search aborted due to worker initialization/read failure: "
                          << worker_failure_msg << std::endl;
            diskann::aligned_free(query);
            if (gt_ids != nullptr) delete[] gt_ids;
            if (gt_dists != nullptr) delete[] gt_dists;
            return -1;
        }
        auto e = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = e - s;
        double qps = (1.0 * query_num) / (1.0 * diff.count());

        diskann::convert_types<uint64_t, uint32_t>(query_result_ids_64.data(), query_result_ids[test_id].data(),
                                                   query_num, recall_at);

        auto mean_latency = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto latency_999 = diskann::get_percentile_stats<float>(
            stats, query_num, 0.999, [](const diskann::QueryStats &stats) { return stats.total_us; });

        auto mean_ios = diskann::get_mean_stats<uint32_t>(stats, query_num,
                                                          [](const diskann::QueryStats &stats) { return stats.n_ios; });

        auto mean_graph_ios = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_graph_ios; });

        auto mean_refined_vecs = diskann::get_mean_stats<float>(
            stats, query_num,
            [](const diskann::QueryStats &stats) { return static_cast<float>(stats.n_ios - stats.n_graph_ios); });

        auto mean_cache_hits = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_cache_hits; });

        auto mean_beam_rounds = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_beam_rounds; });

        auto mean_beam_no_io_rounds = diskann::get_mean_stats<uint32_t>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.n_beam_no_io_rounds; });

        double total_graph_ios = 0.0;
        double total_cache_hits = 0.0;
        double total_beam_rounds = 0.0;
        double total_beam_no_io_rounds = 0.0;
        double total_graph_io_batches = 0.0;
        for (uint32_t i = 0; i < query_num; i++)
        {
            total_graph_ios += stats[i].n_graph_ios;
            total_cache_hits += stats[i].n_cache_hits;
            total_beam_rounds += stats[i].n_beam_rounds;
            total_beam_no_io_rounds += stats[i].n_beam_no_io_rounds;
            total_graph_io_batches += stats[i].n_graph_io_batches;
        }
        const double cache_hit_rate = (total_graph_ios + total_cache_hits) > 0.0
                                          ? (100.0 * total_cache_hits / (total_graph_ios + total_cache_hits))
                                          : 0.0;
        const double beam_no_io_rate =
            total_beam_rounds > 0.0 ? (100.0 * total_beam_no_io_rounds / total_beam_rounds) : 0.0;
        const double graph_ios_per_beam =
            total_beam_rounds > 0.0 ? (total_graph_ios / total_beam_rounds) : 0.0;
        const double graph_ios_per_io_beam =
            total_graph_io_batches > 0.0 ? (total_graph_ios / total_graph_io_batches) : 0.0;

        auto mean_cpuus = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.cpu_us; });

        auto mean_io_us = diskann::get_mean_stats<float>(stats, query_num,
                                                         [](const diskann::QueryStats &stats) { return stats.io_us; });

        auto mean_traversal_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.traversal_us; });

        auto mean_refinement_us = diskann::get_mean_stats<float>(
            stats, query_num, [](const diskann::QueryStats &stats) { return stats.refinement_us; });


        auto mean_io_batches = diskann::get_mean_stats<float>(stats, query_num,
                 [](const diskann::QueryStats &stats) { return stats.n_io_batches; });

        double recall = 0;
        if (calc_recall_flag)
        {
            recall = diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                               query_result_ids[test_id].data(), recall_at, recall_at);
            best_recall = std::max(recall, best_recall);
        }

        if (query_stats_writer)
        {
            const uint32_t gt_compare_count =
                calc_recall_flag ? static_cast<uint32_t>(std::min<size_t>(gt_dim, recall_at)) : 0;
            for (uint32_t i = 0; i < query_num; i++)
            {
                const auto &qstats = stats[i];
                const unsigned refined_vecs =
                    qstats.n_ios >= qstats.n_graph_ios ? qstats.n_ios - qstats.n_graph_ios : 0;
                double query_recall = 0.0;
                if (calc_recall_flag && gt_compare_count > 0)
                {
                    const uint32_t *gt = gt_ids + static_cast<size_t>(i) * gt_dim;
                    const uint32_t *res = query_result_ids[test_id].data() + static_cast<size_t>(i) * recall_at;
                    uint32_t hits = 0;
                    for (uint32_t r = 0; r < recall_at; r++)
                    {
                        for (uint32_t g = 0; g < gt_compare_count; g++)
                        {
                            if (res[r] == gt[g])
                            {
                                hits++;
                                break;
                            }
                        }
                    }
                    query_recall = static_cast<double>(hits) / static_cast<double>(recall_at);
                }

                query_stats_writer << L << "," << i << "," << qstats.total_us << "," << qstats.io_us << ","
                                   << qstats.cpu_us << "," << qstats.traversal_us << "," << qstats.refinement_us
                                   << "," << qstats.n_ios << "," << qstats.n_graph_ios << "," << refined_vecs
                                   << "," << qstats.n_cache_hits << "," << qstats.n_beam_rounds << ","
                                   << qstats.n_beam_no_io_rounds << "," << qstats.n_io_batches << ","
                                   << qstats.n_graph_io_batches << ",";
                if (calc_recall_flag)
                {
                    query_stats_writer << query_recall;
                }
                query_stats_writer << "\n";
            }
        }

        diskann::cout << std::setw(6) << L << std::setw(12) << effective_beamwidth << std::setw(16) << qps
                      << std::setw(16) << mean_latency << std::setw(16) << latency_999 << std::setw(16) << mean_ios
                      << std::setw(16) << mean_graph_ios << std::setw(16) << mean_cache_hits
                      << std::setw(16) << mean_refined_vecs << std::setw(16) << cache_hit_rate
                      << std::setw(16) << mean_beam_rounds << std::setw(16) << mean_beam_no_io_rounds
                      << std::setw(16) << beam_no_io_rate
                      << std::setw(16) << graph_ios_per_beam << std::setw(16) << graph_ios_per_io_beam
                      << std::setw(16) << mean_io_us << std::setw(16) << mean_cpuus
                      << std::setw(16) << mean_traversal_us << std::setw(16) << mean_refinement_us
                      << std::setw(16) << mean_io_batches;
        if (calc_recall_flag)
        {
            diskann::cout << std::setw(16) << recall << std::endl;
        }
        else
            diskann::cout << std::endl;
        delete[] stats;
    }

    diskann::cout << "Done searching. Now saving results " << std::endl;
    uint64_t test_id = 0;
    for (auto L : Lvec)
    {
        if (L < recall_at)
            continue;

        std::string cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_idx_uint32.bin";
        diskann::save_bin<uint32_t>(cur_result_path, query_result_ids[test_id].data(), query_num, recall_at);

        cur_result_path = result_output_prefix + "_" + std::to_string(L) + "_dists_float.bin";
        diskann::save_bin<float>(cur_result_path, query_result_dists[test_id++].data(), query_num, recall_at);
    }

    diskann::aligned_free(query);
    if (gt_ids != nullptr) delete[] gt_ids;
    if (gt_dists != nullptr) delete[] gt_dists;
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
    std::string data_type, dist_fn, index_path_prefix, result_path_prefix, query_file, gt_file;
    uint32_t num_threads, K, W, num_nodes_to_cache, search_io_limit;
    std::vector<uint32_t> Lvec;
    float fail_if_recall_below = 0.0f;
    bool refine_total = false;
    uint32_t early_exit_streak = 1;
    uint32_t fixed_refine_count = 0;
    double fixed_refine_ratio = 0.0;
    std::string query_stats_path;

    po::options_description desc{
        program_options_utils::make_program_description("sigma_search", "Searches on-disk SIGMA indexes")};

    po::variables_map vm;
    try
    {
        desc.add_options()("help,h", "Print information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path_prefix)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);

        // Optional parameters
        po::options_description optional_configs("Optional");
        optional_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        optional_configs.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(2),
                                       program_options_utils::BEAMWIDTH);
        optional_configs.add_options()("num_nodes_to_cache", po::value<uint32_t>(&num_nodes_to_cache)->default_value(0),
                                       program_options_utils::NUMBER_OF_NODES_TO_CACHE);
        optional_configs.add_options()(
            "search_io_limit",
            po::value<uint32_t>(&search_io_limit)->default_value(std::numeric_limits<uint32_t>::max()),
            "Max #IOs for search.  Default value: uint32::max()");
        optional_configs.add_options()("num_threads,T",
                                       po::value<uint32_t>(&num_threads)->default_value(omp_get_num_procs()),
                                       program_options_utils::NUMBER_THREADS_DESCRIPTION);
        optional_configs.add_options()("fail_if_recall_below",
                                       po::value<float>(&fail_if_recall_below)->default_value(0.0f),
                                       program_options_utils::FAIL_IF_RECALL_BELOW);
        optional_configs.add_options()("start_node", po::value<int64_t>()->default_value(-1),
                                       "Start node for search (Override default). -1 means use default.");
        optional_configs.add_options()("refine_total",
                                       po::bool_switch(&refine_total)->default_value(false),
                                       "Refine every candidate selected for refinement without early termination.");
        optional_configs.add_options()("early_exit_streak",
                                       po::value<uint32_t>(&early_exit_streak)->default_value(1),
                                       "For early-exit mode, stop after the top-k id set is unchanged for this many consecutive refinement batches.");
        optional_configs.add_options()("fixed_refine_count",
                                       po::value<uint32_t>(&fixed_refine_count)->default_value(0),
                                       "Refine at most this many candidates per query and disable dynamic early exit. 0 keeps the existing dynamic/full behavior.");
        optional_configs.add_options()("fixed_refine_ratio",
                                       po::value<double>(&fixed_refine_ratio)->default_value(0.0),
                                       "Refine this fraction of the candidate prefix per query and disable dynamic early exit. 0 keeps the existing dynamic/full behavior.");
        optional_configs.add_options()("query_stats_path",
                                       po::value<std::string>(&query_stats_path)->default_value(std::string("")),
                                       "Write per-query search stats CSV to this path. Empty disables the dump.");

        // Merge required and optional parameters
        desc.add(required_configs).add(optional_configs);

        // po::variables_map vm; // Moved outside
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

    if (data_type != std::string("float"))
    {
        std::cerr << "Unsupported data type for the active HiDANN combined search path. Use --data_type float."
                  << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == std::string("mips"))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else
    {
        std::cout << "Unsupported distance function..." << std::endl;
        return -1;
    }

    int64_t start_node = vm["start_node"].as<int64_t>();

    try
    {
        return search_disk_index<float>(metric, index_path_prefix, result_path_prefix, query_file, gt_file,
                                        num_threads, K, W, num_nodes_to_cache, search_io_limit, Lvec,
                                        fail_if_recall_below, start_node, refine_total, early_exit_streak,
                                        fixed_refine_count, fixed_refine_ratio, query_stats_path);
    }
    catch (const std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        return -1;
    }
}

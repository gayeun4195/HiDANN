// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/program_options.hpp>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace po = boost::program_options;

namespace {

[[noreturn]] void fail(const std::string &msg)
{
    throw std::runtime_error(msg);
}

template <typename T> T read_pod(std::ifstream &in, const std::string &what)
{
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in)
    {
        fail("failed to read " + what);
    }
    return value;
}

void read_exact(std::ifstream &in, char *dst, size_t n, const std::string &what)
{
    in.read(dst, static_cast<std::streamsize>(n));
    if (static_cast<size_t>(in.gcount()) != n || !in)
    {
        fail("failed to read " + what);
    }
}

std::string csv_escape(const std::string &value)
{
    if (value.find_first_of(",\"\n\r") == std::string::npos)
    {
        return value;
    }
    std::string out = "\"";
    for (char ch : value)
    {
        if (ch == '"')
        {
            out += "\"\"";
        }
        else
        {
            out += ch;
        }
    }
    out += '"';
    return out;
}

class MMapFile
{
  public:
    MMapFile() = default;

    explicit MMapFile(const std::string &path)
    {
        open(path);
    }

    MMapFile(const MMapFile &) = delete;
    MMapFile &operator=(const MMapFile &) = delete;

    MMapFile(MMapFile &&other) noexcept
    {
        move_from(other);
    }

    MMapFile &operator=(MMapFile &&other) noexcept
    {
        if (this != &other)
        {
            close();
            move_from(other);
        }
        return *this;
    }

    ~MMapFile()
    {
        close();
    }

    void open(const std::string &path)
    {
        close();
        path_ = path;
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
        {
            fail("cannot open " + path);
        }

        struct stat st
        {
        };
        if (::fstat(fd_, &st) != 0)
        {
            fail("cannot stat " + path);
        }
        if (st.st_size <= 0)
        {
            fail("empty file: " + path);
        }
        length_ = static_cast<size_t>(st.st_size);
        data_ = static_cast<const char *>(::mmap(nullptr, length_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED)
        {
            data_ = nullptr;
            fail("mmap failed for " + path);
        }
#ifdef MADV_RANDOM
        (void)::madvise(const_cast<char *>(data_), length_, MADV_RANDOM);
#endif
    }

    const char *data() const
    {
        return data_;
    }

    size_t size() const
    {
        return length_;
    }

    const std::string &path() const
    {
        return path_;
    }

  private:
    void close()
    {
        if (data_ != nullptr)
        {
            ::munmap(const_cast<char *>(data_), length_);
            data_ = nullptr;
        }
        if (fd_ >= 0)
        {
            ::close(fd_);
            fd_ = -1;
        }
        length_ = 0;
    }

    void move_from(MMapFile &other)
    {
        fd_ = other.fd_;
        length_ = other.length_;
        data_ = other.data_;
        path_ = std::move(other.path_);
        other.fd_ = -1;
        other.length_ = 0;
        other.data_ = nullptr;
    }

    int fd_ = -1;
    size_t length_ = 0;
    const char *data_ = nullptr;
    std::string path_;
};

struct BaseMatrix
{
    MMapFile file;
    uint32_t n = 0;
    uint32_t dim = 0;
    const float *values = nullptr;

    explicit BaseMatrix(const std::string &path) : file(path)
    {
        if (file.size() < 2 * sizeof(uint32_t))
        {
            fail("fbin header is too short: " + path);
        }
        std::memcpy(&n, file.data(), sizeof(uint32_t));
        std::memcpy(&dim, file.data() + sizeof(uint32_t), sizeof(uint32_t));
        const size_t expected = 2 * sizeof(uint32_t) + static_cast<size_t>(n) * dim * sizeof(float);
        if (expected != file.size())
        {
            std::ostringstream oss;
            oss << "unexpected fbin size for " << path << ": got " << file.size() << ", expected " << expected;
            fail(oss.str());
        }
        values = reinterpret_cast<const float *>(file.data() + 2 * sizeof(uint32_t));
    }

    const float *row(uint32_t i) const
    {
        return values + static_cast<size_t>(i) * dim;
    }
};

struct Matrix
{
    uint32_t n = 0;
    uint32_t dim = 0;
    std::vector<float> values;

    const float *row(uint32_t i) const
    {
        return values.data() + static_cast<size_t>(i) * dim;
    }
};

struct GroundTruth
{
    uint32_t n = 0;
    uint32_t width = 0;
    std::vector<uint32_t> ids;
    std::vector<float> dists;
};

struct Graph
{
    std::vector<std::vector<uint32_t>> adj;
    uint32_t entry_point = 0;
};

struct Neighbor
{
    float distance = 0.0f;
    uint32_t id = 0;
    bool expanded = false;
};

struct SearchResult
{
    float recall = 0.0f;
    uint32_t hop = 0;
    uint32_t dist_calcs = 0;
};

struct Aggregate
{
    double recall_sum = 0.0;
    double hop_sum = 0.0;
    double dist_sum = 0.0;
    uint32_t queries = 0;
};

class NeighborQueue
{
  public:
    explicit NeighborQueue(size_t capacity) : capacity_(capacity)
    {
        data_.reserve(capacity + 1);
    }

    void clear()
    {
        data_.clear();
        cur_ = 0;
    }

    void insert(float distance, uint32_t id)
    {
        Neighbor nbr{distance, id, false};
        if (data_.size() == capacity_ && less_neighbor(data_.back(), nbr))
        {
            return;
        }
        for (const auto &x : data_)
        {
            if (x.id == id)
            {
                return;
            }
        }

        size_t lo = 0;
        size_t hi = data_.size();
        while (lo < hi)
        {
            const size_t mid = lo + (hi - lo) / 2;
            if (less_neighbor(nbr, data_[mid]))
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }
        data_.insert(data_.begin() + static_cast<std::ptrdiff_t>(lo), nbr);
        if (data_.size() > capacity_)
        {
            data_.pop_back();
        }
        if (lo < cur_)
        {
            cur_ = lo;
        }
    }

    bool has_unexpanded_node() const
    {
        return cur_ < data_.size();
    }

    Neighbor closest_unexpanded()
    {
        data_[cur_].expanded = true;
        Neighbor res = data_[cur_];
        ++cur_;
        while (cur_ < data_.size() && data_[cur_].expanded)
        {
            ++cur_;
        }
        return res;
    }

    size_t size() const
    {
        return data_.size();
    }

    const Neighbor &operator[](size_t i) const
    {
        return data_[i];
    }

  private:
    static bool less_neighbor(const Neighbor &a, const Neighbor &b)
    {
        return a.distance < b.distance;
    }

    size_t capacity_;
    std::vector<Neighbor> data_;
    size_t cur_ = 0;
};

class SearchScratch
{
  public:
    explicit SearchScratch(uint32_t n) : seen_(n, 0)
    {
    }

    void next_query()
    {
        ++token_;
        if (token_ == 0)
        {
            std::fill(seen_.begin(), seen_.end(), 0);
            token_ = 1;
        }
    }

    bool mark_if_new(uint32_t id)
    {
        if (seen_[id] == token_)
        {
            return false;
        }
        seen_[id] = token_;
        return true;
    }

  private:
    std::vector<uint32_t> seen_;
    uint32_t token_ = 1;
};

Matrix read_fbin(const std::string &path, uint32_t max_points = 0)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail("cannot open " + path);
    }
    Matrix m;
    m.n = read_pod<uint32_t>(in, "fbin n");
    m.dim = read_pod<uint32_t>(in, "fbin dim");
    if (max_points > 0 && max_points < m.n)
    {
        m.n = max_points;
    }
    const size_t count = static_cast<size_t>(m.n) * m.dim;
    m.values.resize(count);
    read_exact(in, reinterpret_cast<char *>(m.values.data()), count * sizeof(float), "fbin payload");
    return m;
}

GroundTruth read_ground_truth(const std::string &path, uint32_t max_queries)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail("cannot open " + path);
    }
    GroundTruth gt;
    gt.n = read_pod<uint32_t>(in, "gt n");
    gt.width = read_pod<uint32_t>(in, "gt width");
    const uint32_t read_n = max_queries > 0 && max_queries < gt.n ? max_queries : gt.n;
    const size_t ids_count = static_cast<size_t>(read_n) * gt.width;
    gt.ids.resize(ids_count);
    read_exact(in, reinterpret_cast<char *>(gt.ids.data()), ids_count * sizeof(uint32_t), "gt ids");

    const size_t full_ids_bytes = static_cast<size_t>(gt.n) * gt.width * sizeof(uint32_t);
    const size_t read_ids_bytes = ids_count * sizeof(uint32_t);
    if (full_ids_bytes > read_ids_bytes)
    {
        in.seekg(static_cast<std::streamoff>(full_ids_bytes - read_ids_bytes), std::ios::cur);
    }

    gt.dists.resize(ids_count);
    read_exact(in, reinterpret_cast<char *>(gt.dists.data()), ids_count * sizeof(float), "gt distances");
    gt.n = read_n;
    return gt;
}

Graph load_mem_index(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail("cannot open memory index " + path);
    }

    (void)read_pod<uint64_t>(in, "mem index size");
    (void)read_pod<uint32_t>(in, "mem max observed degree");
    Graph graph;
    graph.entry_point = read_pod<uint32_t>(in, "mem entry point");
    (void)read_pod<uint64_t>(in, "mem frozen count");

    while (true)
    {
        uint32_t deg = 0;
        in.read(reinterpret_cast<char *>(&deg), sizeof(uint32_t));
        if (!in)
        {
            if (in.eof())
            {
                break;
            }
            fail("truncated degree in " + path);
        }
        std::vector<uint32_t> nbrs(deg);
        if (deg > 0)
        {
            read_exact(in, reinterpret_cast<char *>(nbrs.data()), static_cast<size_t>(deg) * sizeof(uint32_t),
                       "mem neighbors");
        }
        graph.adj.push_back(std::move(nbrs));
    }
    return graph;
}

float l2_distance(const float *x, const float *q, uint32_t dim)
{
    float acc = 0.0f;
    for (uint32_t d = 0; d < dim; ++d)
    {
        const float diff = x[d] - q[d];
        acc += diff * diff;
    }
    return acc;
}

float calculate_recall(const GroundTruth &gt, uint32_t qi, const std::vector<uint32_t> &res, uint32_t k)
{
    if (gt.width < k)
    {
        fail("groundtruth width is smaller than recall cutoff");
    }
    const uint32_t *ids = gt.ids.data() + static_cast<size_t>(qi) * gt.width;
    const float *dists = gt.dists.data() + static_cast<size_t>(qi) * gt.width;
    const float cutoff = dists[k - 1];

    uint32_t tie = k;
    while (tie < gt.width && dists[tie] == cutoff)
    {
        ++tie;
    }

    uint32_t hits = 0;
    for (uint32_t i = 0; i < tie; ++i)
    {
        const uint32_t gt_id = ids[i];
        for (uint32_t j = 0; j < k && j < res.size(); ++j)
        {
            if (res[j] == gt_id)
            {
                ++hits;
                break;
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(k);
}

SearchResult evaluate_one_query(const BaseMatrix &data, const Graph &graph, const Matrix &queries,
                                const GroundTruth &gt, uint32_t qi, uint32_t l_value, uint32_t k,
                                SearchScratch &scratch, uint32_t start_node)
{
    if (l_value < k)
    {
        fail("L must be greater than or equal to recall cutoff");
    }
    if (start_node >= graph.adj.size())
    {
        fail("search start node is outside graph");
    }

    scratch.next_query();
    NeighborQueue queue(l_value);
    std::vector<uint32_t> expanded_order;
    expanded_order.reserve(static_cast<size_t>(l_value) + 16);

    const float *q = queries.row(qi);
    uint32_t dist_calcs = 0;

    scratch.mark_if_new(start_node);
    queue.insert(l2_distance(data.row(start_node), q, data.dim), start_node);
    ++dist_calcs;

    while (queue.has_unexpanded_node())
    {
        const Neighbor nb = queue.closest_unexpanded();
        const uint32_t cur = nb.id;
        expanded_order.push_back(cur);

        for (uint32_t u : graph.adj[cur])
        {
            if (u >= graph.adj.size())
            {
                continue;
            }
            if (!scratch.mark_if_new(u))
            {
                continue;
            }
            queue.insert(l2_distance(data.row(u), q, data.dim), u);
            ++dist_calcs;
        }
    }

    std::vector<uint32_t> result;
    const size_t kk = std::min<size_t>(k, queue.size());
    result.reserve(kk);
    for (size_t i = 0; i < kk; ++i)
    {
        result.push_back(queue[i].id);
    }

    SearchResult out;
    out.recall = calculate_recall(gt, qi, result, k);
    out.hop = static_cast<uint32_t>(expanded_order.size());
    out.dist_calcs = dist_calcs;
    return out;
}

Aggregate evaluate_l(const BaseMatrix &data, const Graph &graph, const Matrix &queries, const GroundTruth &gt,
                     uint32_t l_value, uint32_t k, uint32_t threads, uint32_t start_node)
{
    Aggregate aggregate;
    aggregate.queries = queries.n;
    std::atomic<uint32_t> next{0};
    std::mutex aggregate_mutex;
    const uint32_t actual_threads = std::max<uint32_t>(1, threads);
    std::vector<std::thread> workers;
    workers.reserve(actual_threads);

    for (uint32_t t = 0; t < actual_threads; ++t)
    {
        workers.emplace_back([&]() {
            SearchScratch scratch(static_cast<uint32_t>(graph.adj.size()));
            Aggregate local;
            while (true)
            {
                const uint32_t qi = next.fetch_add(1, std::memory_order_relaxed);
                if (qi >= queries.n)
                {
                    break;
                }
                const SearchResult result =
                    evaluate_one_query(data, graph, queries, gt, qi, l_value, k, scratch, start_node);
                local.recall_sum += result.recall;
                local.hop_sum += result.hop;
                local.dist_sum += result.dist_calcs;
            }

            std::lock_guard<std::mutex> lock(aggregate_mutex);
            aggregate.recall_sum += local.recall_sum;
            aggregate.hop_sum += local.hop_sum;
            aggregate.dist_sum += local.dist_sum;
        });
    }

    for (auto &worker : workers)
    {
        worker.join();
    }
    return aggregate;
}

std::string format_seconds(double secs)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2) << secs << "s";
    return oss.str();
}

int run(int argc, char **argv)
{
    std::string data_type;
    std::string dist_fn;
    std::string base_file;
    std::string index_path;
    std::string query_file;
    std::string gt_file;
    std::string dataset;
    std::string method;
    std::string method_id;
    std::string entry_protocol;
    std::string metric_source;
    std::string out_csv;
    int64_t start_node = -1;
    uint32_t num_threads = std::max<uint32_t>(1, std::thread::hardware_concurrency());
    uint32_t recall_at = 10;
    uint32_t max_queries = 0;
    std::vector<uint32_t> l_values;

    po::options_description desc("evaluate_memory_index_quality options");
    desc.add_options()("help,h", "print help")(
        "data_type", po::value<std::string>(&data_type)->default_value("float"), "data type; only float is supported")(
        "dist_fn", po::value<std::string>(&dist_fn)->default_value("l2"), "distance; only l2 is supported")(
        "base_file", po::value<std::string>(&base_file)->required(), "base vectors in fbin format")(
        "index_path_prefix", po::value<std::string>(&index_path)->required(), "memory index graph path")(
        "query_file", po::value<std::string>(&query_file)->required(), "query vectors in fbin format")(
        "gt_file", po::value<std::string>(&gt_file)->required(), "ground-truth file")(
        "search_list,L", po::value<std::vector<uint32_t>>(&l_values)->multitoken()->required(), "search L values")(
        "num_threads,T", po::value<uint32_t>(&num_threads)->default_value(num_threads), "evaluation threads")(
        "recall_at,K", po::value<uint32_t>(&recall_at)->default_value(10), "recall cutoff")(
        "max_queries", po::value<uint32_t>(&max_queries)->default_value(0), "optional query-count cap")(
        "dataset", po::value<std::string>(&dataset)->default_value("simplewiki"), "dataset id")(
        "method", po::value<std::string>(&method)->required(), "display method name")(
        "method_id", po::value<std::string>(&method_id)->required(), "plot method id")(
        "entry_protocol", po::value<std::string>(&entry_protocol)->default_value("single_common_entry"),
        "entry-node protocol label")(
        "start_node", po::value<int64_t>(&start_node)->required(), "fixed common entry node id")(
        "metric_source", po::value<std::string>(&metric_source)->default_value("memory_index_hop"),
        "measurement provenance label")("out_csv", po::value<std::string>(&out_csv)->required(), "output CSV");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return 0;
    }
    po::notify(vm);

    if (data_type != "float")
    {
        fail("only float data_type is supported by the graph-direct quality evaluator");
    }
    if (dist_fn != "l2")
    {
        fail("only l2 distance is supported by the graph-direct quality evaluator");
    }
    if (l_values.empty())
    {
        fail("search_list must contain at least one L value");
    }
    if (start_node < 0)
    {
        fail("start_node must be non-negative for single-common-entry evaluation");
    }

    const uint32_t max_l = *std::max_element(l_values.begin(), l_values.end());
    if (max_l < recall_at)
    {
        fail("all L values are smaller than recall_at");
    }

    std::cout << "Loading base vectors: " << base_file << "\n";
    BaseMatrix data(base_file);
    std::cout << "  N=" << data.n << " D=" << data.dim << " mapped_bytes=" << data.file.size() << "\n";

    std::cout << "Loading queries: " << query_file << "\n";
    Matrix queries = read_fbin(query_file, max_queries);
    if (queries.dim != data.dim)
    {
        fail("query dimension does not match base dimension");
    }
    std::cout << "  Q=" << queries.n << " D=" << queries.dim << "\n";

    std::cout << "Loading ground truth: " << gt_file << "\n";
    GroundTruth gt = read_ground_truth(gt_file, queries.n);
    if (gt.n != queries.n)
    {
        fail("ground-truth query count does not match loaded query count");
    }
    if (gt.width < recall_at)
    {
        fail("ground-truth width is smaller than recall_at");
    }
    std::cout << "  Q=" << gt.n << " width=" << gt.width << "\n";

    std::cout << "Loading memory graph: " << index_path << "\n";
    Graph graph = load_mem_index(index_path);
    if (graph.adj.size() != data.n)
    {
        std::ostringstream oss;
        oss << "index has " << graph.adj.size() << " nodes, base data has " << data.n << " rows";
        fail(oss.str());
    }
    if (static_cast<uint64_t>(start_node) >= graph.adj.size())
    {
        fail("start_node is outside the loaded graph");
    }
    std::cout << "  nodes=" << graph.adj.size() << " native_entry=" << graph.entry_point
              << " common_entry=" << start_node << "\n";
    std::cout << "dataset=" << dataset << "\n";
    std::cout << "method_id=" << method_id << "\n";
    std::cout << "entry_protocol=" << entry_protocol << "\n";
    std::cout << "threads=" << num_threads << " recall_at=" << recall_at << "\n";
    std::cout << "L,recall_at_10,one_minus_recall_at_10,avg_hop,avg_cmps,qps\n";

    std::ofstream out(out_csv);
    if (!out)
    {
        fail("could not open output CSV: " + out_csv);
    }
    out << "dataset,method,method_id,L,recall_at_10,one_minus_recall_at_10,avg_hop,avg_cmps,qps,"
           "threads,entry_protocol,beam_width,start_node,metric_source,index_path,source\n";
    out << std::fixed << std::setprecision(6);

    for (uint32_t l_value : l_values)
    {
        if (l_value < recall_at)
        {
            std::cerr << "skipping L=" << l_value << " because it is smaller than recall_at=" << recall_at << "\n";
            continue;
        }
        const auto start = std::chrono::steady_clock::now();
        Aggregate aggregate =
            evaluate_l(data, graph, queries, gt, l_value, recall_at, num_threads, static_cast<uint32_t>(start_node));
        const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
        const double denom = static_cast<double>(std::max<uint32_t>(1, aggregate.queries));
        const double recall = aggregate.recall_sum / denom;
        const double error = std::max(1.0 - recall, 1e-9);
        const double avg_hop = aggregate.hop_sum / denom;
        const double avg_cmps = aggregate.dist_sum / denom;
        const double qps = denom / elapsed.count();

        std::cout << l_value << "," << recall << "," << error << "," << avg_hop << "," << avg_cmps << "," << qps
                  << " elapsed=" << format_seconds(elapsed.count()) << "\n";
        out << csv_escape(dataset) << "," << csv_escape(method) << "," << csv_escape(method_id) << "," << l_value
            << "," << recall << "," << error << "," << avg_hop << "," << avg_cmps << "," << qps << ","
            << num_threads << "," << csv_escape(entry_protocol) << ",1," << start_node << ","
            << csv_escape(metric_source) << "," << csv_escape(index_path) << "," << csv_escape(out_csv) << "\n";
    }
    return 0;
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        return run(argc, argv);
    }
    catch (const std::exception &e)
    {
        std::cerr << "evaluate_memory_index_quality failed: " << e.what() << "\n";
        return 2;
    }
}

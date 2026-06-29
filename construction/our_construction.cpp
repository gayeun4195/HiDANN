/*
 * our_construction.cpp
 *
 * Purpose:
 *   Main partitioned graph construction logic.
 *   1. Splits data into partitions.
 *   2. Builds sub-indices (DiskANN) for each partition.
 *   3. Performs cross-partition search and occlusion pruning to find neighbors across boundaries.
 *   4. Merges edges into per-bucket logs and compacts them into a candidate graph (cand.idx/cand.ids).
 *
 * Usage:
 *   ./our_construction <data.fbin> <out_prefix> [P] [R] [L] [threads] ...
 */

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

#include <omp.h>
#include <time.h>

#include "index.h"
#include "index_factory.h"
#include "parameters.h"
#include "utils.h"

using T = float;
using Clock = std::chrono::steady_clock;

static inline double sec(const Clock::time_point &a, const Clock::time_point &b)
{
    return std::chrono::duration<double>(b - a).count();
}
static inline void die(const std::string &m)
{
    std::cerr << "fatal: " << m << "\n";
    std::exit(2);
}
static inline bool env_flag_enabled(const char *name, bool default_value = true)
{
    const char *value = std::getenv(name);
    if (value == nullptr || value[0] == '\0')
        return default_value;
    return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 && std::strcmp(value, "False") != 0 &&
           std::strcmp(value, "no") != 0 && std::strcmp(value, "off") != 0;
}
static inline bool construction_fadvise_enabled()
{
    static const bool enabled = env_flag_enabled("HIDANN_CONSTRUCTION_FADVISE", true);
    return enabled;
}
static inline bool construction_sync_before_drop_enabled()
{
    static const bool enabled = env_flag_enabled("HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP", true);
    return enabled;
}
static inline void trim_released_heap()
{
#if defined(__GLIBC__)
    malloc_trim(0);
#endif
}
static inline void drop_file_cache_if_enabled(const std::string &path)
{
    if (!construction_fadvise_enabled())
        return;
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        return;
    ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
}
static inline void drop_index_file_cache_if_enabled(const std::string &prefix)
{
    drop_file_cache_if_enabled(prefix);
    drop_file_cache_if_enabled(prefix + ".data");
}
static inline bool path_exists(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    return (bool)f;
}
static inline void _read_exact(std::ifstream &in, void *dst, size_t n)
{
    in.read(reinterpret_cast<char *>(dst), (std::streamsize)n);
    if (!in)
        die("short read");
}
static inline uint64_t round_up(uint64_t x, uint64_t a)
{
    return (x + a - 1) / a * a;
}

static inline std::string subindex_prefix(const std::string &out_prefix, uint32_t u)
{
    std::ostringstream o;
    o << out_prefix << ".subgraph_u" << u << ".mem.index";
    return o.str();
}
static inline std::string subshard_path(const std::string &out_prefix, uint32_t u)
{
    std::ostringstream o;
    o << out_prefix << ".subshard_u" << u << ".fbin";
    return o.str();
}
static inline std::string bucket_log_name(const std::string &tag_prefix, uint32_t b)
{
    std::ostringstream o;
    o << tag_prefix << "bucket_" << std::setw(3) << std::setfill('0') << b << ".log";
    return o.str();
}

// ---------- CLI Arguments ----------
struct Args
{
    std::string data_path;
    std::string out_prefix;
    uint32_t P = 20, R = 32, L = 50;
    uint32_t threads = 0;

    // All intermediates & finals will include this prefix; default = out_prefix + "."
    std::string tag_prefix = "";

    // Bucket config (choose one):
    int buckets = -1;      // --buckets=N (fixed)
    int read_buf_mib = 2; // reserved scratch (and O_DIRECT bounce if enabled)

    // Subindex retention
    int keep_subindex = 0; // --keep-subindex=0|1 (default 0 -> delete at end)

    // Profiling verbosity
    enum class Prof
    {
        NONE,
        BASIC,
        DETAIL
    } profile = Prof::BASIC; // --profile=

    enum class Stage
    {
        ALL,
        BUILD_SUBGRAPHS,
        PAIR_PROCESSING,
        SELF_DUMP,
        CROSS_PAIRS,
        COMPACT
    } stage = Stage::ALL; // --stage=

    uint64_t chunk_bytes = 0;         // --chunk-bytes=
    uint64_t memory_budget_bytes = 0; // --memory-budget-bytes=
    int64_t anchor_start = -1;        // --anchor-start= / --anchor=
    int64_t anchor_end = -1;          // --anchor-end= / --anchor=
    int64_t target = -1;              // --target=, only with --stage=cross-pairs and one --anchor
    std::string cross_pairs_file = ""; // --cross-pairs-file=
};

static inline const char *stage_name(Args::Stage s)
{
    switch (s)
    {
    case Args::Stage::ALL:
        return "all";
    case Args::Stage::BUILD_SUBGRAPHS:
        return "build-subgraphs";
    case Args::Stage::PAIR_PROCESSING:
        return "pair-processing";
    case Args::Stage::SELF_DUMP:
        return "self-dump";
    case Args::Stage::CROSS_PAIRS:
        return "cross-pairs";
    case Args::Stage::COMPACT:
        return "compact";
    }
    return "unknown";
}

static inline void parse_flags(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; i++)
    {
        const char *s = argv[i];
        if (std::strncmp(s, "--prefix=", 9) == 0)
        {
            a.tag_prefix = std::string(s + 9);
        }
        else if (std::strncmp(s, "--buckets=", 10) == 0)
        {
            a.buckets = std::atoi(s + 10);
        }
        else if (std::strncmp(s, "--read-buf-mib=", 15) == 0)
        {
            a.read_buf_mib = std::atoi(s + 15);
        }
        else if (std::strncmp(s, "--keep-subindex=", 16) == 0)
        {
            a.keep_subindex = std::atoi(s + 16);
        }
        else if (std::strncmp(s, "--profile=", 10) == 0)
        {
            std::string v(s + 10);
            if (v == "none")
                a.profile = Args::Prof::NONE;
            else if (v == "detail")
                a.profile = Args::Prof::DETAIL;
            else
                a.profile = Args::Prof::BASIC;
        }
        else if (std::strncmp(s, "--stage=", 8) == 0)
        {
            std::string v(s + 8);
            if (v == "all")
                a.stage = Args::Stage::ALL;
            else if (v == "build-subgraphs")
                a.stage = Args::Stage::BUILD_SUBGRAPHS;
            else if (v == "pair-processing")
                a.stage = Args::Stage::PAIR_PROCESSING;
            else if (v == "self-dump")
                a.stage = Args::Stage::SELF_DUMP;
            else if (v == "cross-pairs")
                a.stage = Args::Stage::CROSS_PAIRS;
            else if (v == "compact")
                a.stage = Args::Stage::COMPACT;
            else
                die("unknown --stage value: " + v);
        }
        else if (std::strncmp(s, "--chunk-bytes=", 14) == 0)
        {
            a.chunk_bytes = std::stoull(s + 14);
        }
        else if (std::strncmp(s, "--memory-budget-bytes=", 22) == 0)
        {
            a.memory_budget_bytes = std::stoull(s + 22);
        }
        else if (std::strncmp(s, "--anchor=", 9) == 0)
        {
            a.anchor_start = std::stoll(s + 9);
            a.anchor_end = a.anchor_start;
        }
        else if (std::strncmp(s, "--anchor-start=", 15) == 0)
        {
            a.anchor_start = std::stoll(s + 15);
        }
        else if (std::strncmp(s, "--anchor-end=", 13) == 0)
        {
            a.anchor_end = std::stoll(s + 13);
        }
        else if (std::strncmp(s, "--target=", 9) == 0)
        {
            a.target = std::stoll(s + 9);
        }
        else if (std::strncmp(s, "--cross-pairs-file=", 19) == 0)
        {
            a.cross_pairs_file = std::string(s + 19);
        }
    }
}

struct CrossPairMask
{
    bool enabled = false;
    uint64_t selected_pairs = 0;
    std::vector<uint8_t> selected;
};

static inline uint64_t unordered_pair_count(uint32_t P)
{
    return (P < 2) ? 0ull : ((uint64_t)P * (uint64_t)(P - 1)) / 2ull;
}

static CrossPairMask load_cross_pair_mask(const Args &a)
{
    CrossPairMask mask;
    if (a.cross_pairs_file.empty())
        return mask;

    mask.enabled = true;
    mask.selected.assign((size_t)a.P * (size_t)a.P, 0);

    std::ifstream in(a.cross_pairs_file);
    if (!in)
        die("open cross pairs file: " + a.cross_pairs_file);

    std::string line;
    uint64_t line_no = 0;
    while (std::getline(in, line))
    {
        ++line_no;
        if (line.empty() || line[0] == '#')
            continue;

        std::istringstream ss(line);
        uint64_t x = 0, y = 0;
        if (!(ss >> x >> y))
        {
            std::ostringstream msg;
            msg << "bad cross pairs file line " << line_no << ": " << line;
            die(msg.str());
        }
        if (x >= a.P || y >= a.P || x == y)
        {
            std::ostringstream msg;
            msg << "invalid cross pair at line " << line_no << ": (" << x << "," << y << ") for P=" << a.P;
            die(msg.str());
        }

        uint32_t i = (uint32_t)std::min<uint64_t>(x, y);
        uint32_t j = (uint32_t)std::max<uint64_t>(x, y);
        size_t idx = (size_t)i * (size_t)a.P + (size_t)j;
        if (!mask.selected[idx])
        {
            mask.selected[idx] = 1;
            ++mask.selected_pairs;
        }
    }

    return mask;
}

static inline bool cross_pair_selected(const CrossPairMask &mask, uint32_t P, uint32_t i, uint32_t j)
{
    if (!mask.enabled)
        return true;
    if (i == j)
        return false;
    if (i > j)
        std::swap(i, j);
    return mask.selected[(size_t)i * (size_t)P + (size_t)j] != 0;
}
static Args::Prof g_prof = Args::Prof::BASIC;
static inline bool prof_basic()
{
    return g_prof == Args::Prof::BASIC || g_prof == Args::Prof::DETAIL;
}


// ---------- Utility Functions ----------
static inline uint64_t thread_cpu_ns()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------- Load Subgraph ----------
static bool load_graph_to_adj_memindex(const std::string &prefix, uint64_t tsize,
                                       std::vector<std::vector<uint32_t>> &adj, uint64_t &max_deg, uint64_t &edge_cnt)
{
    std::ifstream f(prefix, std::ios::binary);
    if (!f)
        return false;
    uint64_t index_size = 0;
    uint32_t maxObserved = 0, entry_point = 0;
    uint64_t num_frozen = 0;
    _read_exact(f, &index_size, 8);
    _read_exact(f, &maxObserved, 4);
    _read_exact(f, &entry_point, 4);
    _read_exact(f, &num_frozen, 8);
    adj.clear();
    adj.reserve((size_t)tsize);
    max_deg = 0;
    edge_cnt = 0;
    uint64_t vid = 0;
    while (true)
    {
        uint32_t deg = 0;
        f.read(reinterpret_cast<char *>(&deg), 4);
        if (!f)
            break;
        std::vector<uint32_t> ids(deg);
        if (deg)
            _read_exact(f, ids.data(), (size_t)deg * 4);
        max_deg = std::max<uint64_t>(max_deg, deg);
        edge_cnt += deg;
        adj.emplace_back(std::move(ids));
        ++vid;
    }
    return vid == tsize;
}

static inline std::vector<uint64_t> make_offsets(uint64_t n, uint32_t P)
{
    std::vector<uint64_t> off(P, 0);
    uint64_t base = n / P, rem = n % P, s = 0;
    for (uint32_t i = 0; i < P; i++)
    {
        uint64_t sz = base + (i < rem ? 1 : 0);
        off[i] = s;
        s += sz;
    }
    return off;
}

// ---------- Pruning Logic ----------
static inline float l2sqr(const float *a, const float *b, uint32_t D)
{
    double acc = 0.0;
#pragma omp simd reduction(+ : acc)
    for (uint32_t k = 0; k < D; k++)
    {
        double t = (double)a[k] - (double)b[k];
        acc += t * t;
    }
    return (float)acc;
}
struct CN
{
    uint32_t id;
    float distance;
    inline bool operator<(const CN &o) const
    {
        return (distance < o.distance) || (distance == o.distance && id < o.id);
    }
};

struct PairAccessorPtr
{
    diskann::Index<T> *idx_i = nullptr;
    uint32_t is = 0, ie = 0;
    diskann::Index<T> *idx_j = nullptr;
    uint32_t js = 0, je = 0;
    uint32_t D = 0;
    inline const T *get(uint32_t gid) const
    {
        if (gid >= is && gid < ie)
            return idx_i->data_ptr_local(gid - is);
        if (gid >= js && gid < je)
            return idx_j->data_ptr_local(gid - js);
        return nullptr;
    }

    inline void load_self_neighbors_and_d2(uint32_t local_i, const float *q, std::vector<uint32_t> &out_ids,
                                           std::vector<float> &out_d2) const
    {
        out_ids.clear();
        out_d2.clear();

        static thread_local std::vector<uint32_t> nbrs;

        nbrs.clear();
        idx_i->copy_neighbors_local(local_i, nbrs);

        out_ids.resize(nbrs.size());
        out_d2.resize(nbrs.size());

        const uint32_t is32 = is;
        for (size_t t = 0; t < nbrs.size(); ++t)
        {
            uint32_t lid = nbrs[t];
            uint32_t gid = is32 + lid;
            out_ids[t] = gid;
            const float *x = idx_i->data_ptr_local(lid);
            out_d2[t] = l2sqr(q, x, D);
        }
    }

    inline void load_self_neighbors_and_d2_j(uint32_t local_j, const float *q, std::vector<uint32_t> &out_ids,
                                             std::vector<float> &out_d2) const
    {
        out_ids.clear();
        out_d2.clear();

        static thread_local std::vector<uint32_t> nbrs;

        nbrs.clear();
        idx_j->copy_neighbors_local(local_j, nbrs);

        out_ids.resize(nbrs.size());
        out_d2.resize(nbrs.size());

        const uint32_t js32 = js;
        for (size_t t = 0; t < nbrs.size(); ++t)
        {
            uint32_t lid = nbrs[t];
            uint32_t gid = js32 + lid;
            out_ids[t] = gid;
            const float *x = idx_j->data_ptr_local(lid);
            out_d2[t] = l2sqr(q, x, D);
        }
    }
};
struct Arena
{
    uint8_t *base = nullptr;
    size_t cap = 0, off = 0;
    ~Arena()
    {
        if (base)
            std::free(base);
    }
    void ensure(size_t need, size_t align = 64)
    {
        if (cap >= need)
            return;
        size_t min_cap = 16 * 1024;
        size_t grow = std::max(need, std::max(min_cap, cap + cap / 2)); // 1.5x
        void *p = nullptr;
        if (posix_memalign(&p, align, grow) != 0)
            throw std::bad_alloc();
        if (base)
            std::free(base);
        base = (uint8_t *)p;
        cap = grow;
        off = 0;
    }
    inline void reset()
    {
        off = 0;
    }
    inline void *alloc_raw(size_t n, size_t a = 64)
    {
        size_t p = (off + (a - 1)) & ~(a - 1);
        if (p + n > cap)
            throw std::bad_alloc();
        off = p + n;
        return base + p;
    }
    template <typename U> inline U *alloc_n(size_t n)
    {
        return reinterpret_cast<U *>(alloc_raw(n * sizeof(U), std::max<size_t>(64, alignof(U))));
    }
};
static thread_local Arena tls_arena;

static void occlude_list_ptr(uint32_t location, CN *pool, size_t pool_sz, float alpha, uint32_t degree,
                             std::vector<uint32_t> &out, const PairAccessorPtr &acc)
{
    if (pool_sz == 0)
        return;
    if (pool_sz > 1)
        std::sort(pool, pool + pool_sz);
    float *ocf = tls_arena.alloc_n<float>(pool_sz);
    for (size_t i = 0; i < pool_sz; i++)
        ocf[i] = 0.f;

    float cur_alpha = 1.f;
    while (cur_alpha <= alpha && out.size() < degree)
    {
        for (size_t t = 0; out.size() < degree && t < pool_sz; t++)
        {
            if (ocf[t] > cur_alpha)
                continue;
            uint32_t vid = pool[t].id;
            if (vid == location)
            {
                ocf[t] = std::numeric_limits<float>::max();
                continue;
            }
            const float *pv = acc.get(vid);
            if (!pv)
            {
                ocf[t] = std::numeric_limits<float>::max();
                continue;
            }
            bool blocked = false;
            float of = 0.f;
            for (uint32_t u : out)
            {
                const float *pu = acc.get(u);
                if (!pu)
                    continue;
                float d_uv = l2sqr(pu, pv, acc.D);
                if (d_uv <= cur_alpha * pool[t].distance)
                {
                    blocked = true;
                    break;
                }
                const float denom = (d_uv <= 0.f) ? 1e-30f : d_uv;
                of = std::max(of, pool[t].distance / denom);
                if (of > cur_alpha)
                {
                    blocked = true;
                    break;
                }
            }
            ocf[t] = blocked ? std::numeric_limits<float>::max() : of;
            if (!blocked)
                out.push_back(vid);
        }
        cur_alpha += 1.f;
    }
}

// ---------- Bucket Log Logic ----------
struct RecHdr
{
    uint32_t gid;
    uint16_t len;
    uint16_t flags;
    uint32_t size_bytes;
}; // 12B
static_assert(sizeof(RecHdr) == 12, "RecHdr");

static constexpr uint32_t REC_HDR_ALIGNED = 16;

// ---------- Append-only Bucket Writer (TLS) ----------
struct TLStaging
{
    int cur_b = -1;
    int cur_fd = -1;
    size_t cap = 0, used = 0;
    uint8_t *buf = nullptr;
    ~TLStaging()
    {
        if (buf)
            std::free(buf);
    }
    void init(size_t bytes)
    {
        if (buf)
            std::free(buf);
        if (bytes < 4096)
            bytes = 4096;
        void *p = nullptr;
        if (posix_memalign(&p, 4096, ((bytes + 4095) / 4096) * 4096) != 0)
            die("posix_memalign staging");
        buf = (uint8_t *)p;
        cap = ((bytes + 4095) / 4096) * 4096;
        used = 0;
        cur_b = -1;
        cur_fd = -1; // NEW
    }
    void flush()
    {
        if (!buf || used == 0)
            return;
        if (cur_fd < 0)
            die("flush with invalid cur_fd");
        ssize_t n = ::write(cur_fd, buf, used);
        if (n != (ssize_t)used)
            die("write bucket staging");
        used = 0;
    }
    void ensure(size_t need)
    {
        if (used + need <= cap)
            return;
        size_t ncap = cap ? (cap * 2) : (64 * 1024);
        while (ncap < used + need)
            ncap *= 2;
        void *p = nullptr;
        if (posix_memalign(&p, 4096, ((ncap + 4095) / 4096) * 4096) != 0)
            die("posix_memalign grow");
        std::memcpy(p, buf, used);
        std::free(buf);
        buf = (uint8_t *)p;
        cap = ((ncap + 4095) / 4096) * 4096;
    }
    void append_record(int fd, int b, uint32_t gid, const uint32_t *ids, uint16_t len)
    {
        if (!len)
            return;
        const uint32_t body = 4u * len;
        const uint32_t sz = (uint32_t)sizeof(RecHdr) + body;
        const uint32_t sz16 = (uint32_t)round_up(sz + 4, REC_HDR_ALIGNED);
        if (cur_b != b)
        {
            if (cur_b >= 0)
                flush();
            cur_b = b;
            cur_fd = fd;
        }
        ensure(sz16);
        RecHdr rh{gid, len, 0u, sz16};
        std::memcpy(buf + used, &rh, sizeof(rh));
        used += sizeof(rh);
        std::memcpy(buf + used, ids, body);
        used += body;
        std::memcpy(buf + used, &sz16, 4);
        used += 4;
        const uint32_t pad = sz16 - (sizeof(RecHdr) + body + 4);
        if (pad)
        {
            std::memset(buf + used, 0, pad);
            used += pad;
        }
        if (used + sz16 > cap)
            flush();
    }
};
static thread_local TLStaging tls_stage;

struct BucketFDs
{
    std::vector<int> fds;
    void open_all(const std::string &tag_prefix, uint32_t B, bool truncate)
    {
        fds.resize(B, -1);
        for (uint32_t b = 0; b < B; b++)
        {
            std::string p = bucket_log_name(tag_prefix, b);
            int flags = O_WRONLY | O_CREAT | O_APPEND;
            if (truncate)
                flags |= O_TRUNC;
            int fd = ::open(p.c_str(), flags, 0644);
            if (fd < 0)
                die("open bucket log failed: " + p);
            fds[b] = fd;
        }
    }
    void close_all()
    {
        for (int &fd : fds)
        {
            if (fd >= 0)
                ::close(fd);
            fd = -1;
        }
    }
};

// ---------- Main ----------
int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " <data.fbin> <out_prefix> [P] [R] [L] [threads]\n"
                  << "  --prefix=TAG_                  # prefix for logs & finals (default: out_prefix + \".\")\n"
                  << "  --buckets=B                    # fixed bucket count (optional)\n"
                  << "  --read-buf-mib=R               # read scratch MiB (default 2)\n"
                  << "  --keep-subindex=0|1            # keep subindex files after done (default 0)\n"
                  << "  --profile=none|basic|detail    # log verbosity\n"
                  << "  --stage=all|build-subgraphs|pair-processing|self-dump|cross-pairs|compact\n"
                  << "  --anchor=N                     # cross-pairs anchor i (optional)\n"
                  << "  --target=N                     # cross-pairs target j; requires one --anchor\n"
                  << "  --anchor-start=A --anchor-end=B # inclusive cross-pairs anchor range\n"
                  << "  --cross-pairs-file=PATH        # optional unordered i j pair allowlist\n"
                  << "  --chunk-bytes=N                # dynamic chunk budget bytes\n"
                  << "  --memory-budget-bytes=N        # logical build memory budget bytes\n";
        return 1;
    }
    Args a;
    a.data_path = argv[1];
    a.out_prefix = argv[2];
    if (argc > 3)
        a.P = (uint32_t)std::stoul(argv[3]);
    if (argc > 4)
        a.R = (uint32_t)std::stoul(argv[4]);
    if (argc > 5)
        a.L = (uint32_t)std::stoul(argv[5]);
    if (argc > 6)
        a.threads = (uint32_t)std::stoul(argv[6]);
    if (a.threads)
        omp_set_num_threads((int)a.threads);
    parse_flags(argc, argv, a);
    if (a.chunk_bytes > 0)
    {
         std::string s = std::to_string(a.chunk_bytes);
         setenv("SIGMA_CHUNK_BYTES", s.c_str(), 1);
         std::cout << "[our_construction] Applied SIGMA_CHUNK_BYTES=" << s << " from --chunk-bytes\n";
    }
    std::cout << "[our_construction] HIDANN_CONSTRUCTION_FADVISE=" << (construction_fadvise_enabled() ? 1 : 0)
              << " HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=" << (construction_sync_before_drop_enabled() ? 1 : 0)
              << "\n";
    g_prof = a.profile;

    if (a.tag_prefix.empty())
    {
        a.tag_prefix = a.out_prefix + ".";
    }
    if (a.target >= 0 && a.stage != Args::Stage::CROSS_PAIRS)
    {
        die("--target requires --stage=cross-pairs");
    }

    uint64_t N = 0, D = 0;
    {
        std::ifstream in(a.data_path, std::ios::binary);
        if (!in)
            die("open data");
        uint32_t n32 = 0, d32 = 0;
        _read_exact(in, &n32, 4);
        _read_exact(in, &d32, 4);
        N = n32;
        D = d32;
    }
    const int T = a.threads ? (int)a.threads : omp_get_max_threads();

    if (prof_basic())
    {
        std::cout << "Start pipeline (bucket-logs, ptr-prune, in-mem compact) N=" << N << " D=" << D << " P=" << a.P
                  << " R=" << a.R << " L=" << a.L << " threads=" << T << " stage=" << stage_name(a.stage)
                  << " prefix=\"" << a.tag_prefix << "\" cross_pairs_file=\""
                  << (a.cross_pairs_file.empty() ? "ALL" : a.cross_pairs_file) << "\"\n";
    }

    auto off = make_offsets(N, a.P);
    const bool builds_subgraphs = (a.stage == Args::Stage::ALL || a.stage == Args::Stage::BUILD_SUBGRAPHS);
    const bool does_self_dump =
        (a.stage == Args::Stage::ALL || a.stage == Args::Stage::PAIR_PROCESSING || a.stage == Args::Stage::SELF_DUMP);
    const bool does_cross_pairs =
        (a.stage == Args::Stage::ALL || a.stage == Args::Stage::PAIR_PROCESSING || a.stage == Args::Stage::CROSS_PAIRS);
    const bool does_compact =
        (a.stage == Args::Stage::ALL || a.stage == Args::Stage::PAIR_PROCESSING || a.stage == Args::Stage::COMPACT);

    if (a.memory_budget_bytes > 0)
    {
        uint64_t max_part = 0;
        for (uint32_t t = 0; t < a.P; t++)
        {
            uint64_t ts = off[t], te = (t + 1 < a.P) ? off[t + 1] : N;
            max_part = std::max<uint64_t>(max_part, te - ts);
        }

        const double one_index_ram =
            diskann::estimate_ram_usage((size_t)max_part, (uint32_t)D, sizeof(T), a.R);
        const long double two_index_ram = 2.0L * (long double)one_index_ram;
        if (prof_basic() && (a.stage == Args::Stage::ALL || a.stage == Args::Stage::BUILD_SUBGRAPHS ||
                             a.stage == Args::Stage::PAIR_PROCESSING))
        {
            std::cout << "[our_construction] Memory Budget Check:\n"
                      << "  budget_bytes=" << a.memory_budget_bytes
                      << " (" << (a.memory_budget_bytes / 1024.0 / 1024.0 / 1024.0) << " GiB)\n"
                      << "  max_partition_size=" << max_part << " / P=" << a.P << "\n"
                      << "  estimated_one_index_ram=" << one_index_ram << " bytes"
                      << " (" << (one_index_ram / 1024.0 / 1024.0 / 1024.0) << " GiB)\n"
                      << "  estimated_two_index_ram=" << (double)two_index_ram << " bytes"
                      << " (" << ((double)two_index_ram / 1024.0 / 1024.0 / 1024.0) << " GiB)\n";
        }
        if (two_index_ram > (long double)a.memory_budget_bytes)
        {
            die("P is too small for the requested memory budget; increase P before construction");
        }
    }

    // ---------- Subindex Construction ----------
    double build_s = 0.0;
    double build_calculation_s = 0.0;
    auto tb0 = Clock::now();
    uint64_t subindex_total_bytes = 0;
    if (builds_subgraphs)
    {
        for (uint32_t t = 0; t < a.P; t++)
        {
            uint64_t ts = off[t], te = (t + 1 < a.P) ? off[t + 1] : N, tsize = te - ts;
            std::string t_pref = subindex_prefix(a.out_prefix, t);
            if (!path_exists(t_pref + ".data"))
            {
                std::vector<float> part(tsize * D);
                {
                    std::ifstream in(a.data_path, std::ios::binary);
                    if (!in)
                        die("open data");
                    uint64_t header = 8, pitch = D * sizeof(T);
                    in.seekg((std::streamoff)(header + ts * pitch));
                    _read_exact(in, part.data(), (size_t)(tsize * pitch));
                }
                std::string t_sub = subshard_path(a.out_prefix, t);
                {
                    std::ofstream out(t_sub, std::ios::binary);
                    if (!out)
                        die("open subshard");
                    uint32_t n32 = (uint32_t)tsize, d32 = (uint32_t)D;
                    out.write((const char *)&n32, 4);
                    out.write((const char *)&d32, 4);
                    out.write((const char *)part.data(), (std::streamsize)(tsize * D * sizeof(T)));
                }
                auto wparams = diskann::IndexWriteParametersBuilder(a.L, a.R)
                                   .with_alpha(1.0f)
                                   .with_saturate_graph(false)
                                   .with_num_threads(omp_get_max_threads())
                                   .build();
                auto cfg = diskann::IndexConfigBuilder()
                               .with_metric(diskann::Metric::L2)
                               .with_dimension((size_t)D)
                               .with_max_points((size_t)tsize)
                               .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                               .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                               .with_data_type(std::string("float"))
                               .is_dynamic_index(false)
                               .with_index_write_params(wparams)
                               .is_enable_tags(false)
                               .is_pq_dist_build(false)
                               .with_num_pq_chunks(0)
                               .is_use_opq(false)
                               .build();
                auto f = diskann::IndexFactory(cfg);
                auto idx = f.create_instance();
                auto filt = diskann::IndexFilterParamsBuilder().build();
                auto build_start = Clock::now();
                idx->build(t_sub.c_str(), (size_t)tsize, filt);
                idx->save(t_pref.c_str());
                auto build_end = Clock::now();
                build_calculation_s += sec(build_start, build_end);
                std::remove(t_sub.c_str());
            }

            // account sizes
            struct stat stA
            {
            }, stB{};
            if (::stat(t_pref.c_str(), &stA) == 0)
                subindex_total_bytes += (uint64_t)stA.st_size;
            if (::stat((t_pref + ".data").c_str(), &stB) == 0)
                subindex_total_bytes += (uint64_t)stB.st_size;
        }
    }
    auto tb1 = Clock::now();
    if (builds_subgraphs)
        build_s += sec(tb0, tb1);
    else
    {
        for (uint32_t t = 0; t < a.P; t++)
        {
            std::string t_pref = subindex_prefix(a.out_prefix, t);
            struct stat stA
            {
            }, stB{};
            if (::stat(t_pref.c_str(), &stA) != 0 || ::stat((t_pref + ".data").c_str(), &stB) != 0)
                die("missing subindex files for requested stage: " + t_pref);
            subindex_total_bytes += (uint64_t)stA.st_size + (uint64_t)stB.st_size;
        }
    }

    if (a.stage == Args::Stage::BUILD_SUBGRAPHS)
    {
        if (prof_basic())
        {
            std::cout << "\n========== Timing ==========\n";
            std::cout << "Sub-index build calculation (s): " << build_calculation_s << "\n";
            std::cout << "Sub-index build total (s): " << build_s << "\n";
            std::cout << "\n========== I/O Summary ==========\n";
            std::cout << "Subindex total bytes (files): " << subindex_total_bytes << "\n";
            std::cout << "Subindex files kept for pair-processing stage.\n";
        }
        return 0;
    }

    // ---------- Bucket Configuration ----------
    const uint64_t Cmax = (uint64_t)a.R * (uint64_t)a.P;
    const uint64_t per_gid_bytes_inmem = 8ull * Cmax + 64ull; // conservative
    const uint64_t reserve_bytes = (8ull << 20);              // +8MiB

    uint32_t B = 256;
    if (a.buckets > 0)
    {
        B = (uint32_t)a.buckets;
    }
    else
    {
        uint64_t M_budget = 0; // bytes
        const uint64_t vec_bytes = N * D * sizeof(T);
        const uint64_t topo_bytes = N * (uint64_t)a.R * 4ull;
        const uint64_t per_part = (vec_bytes + topo_bytes) / (uint64_t)a.P;
        M_budget = 2ull * per_part;
        uint64_t usable = (M_budget > reserve_bytes) ? (M_budget - reserve_bytes) : (uint64_t)1;
        uint64_t Umax = usable / std::max<uint64_t>(per_gid_bytes_inmem, 1ull);
        if (Umax == 0)
            Umax = 1;
        B = (uint32_t)((N + Umax - 1) / Umax);
        if (B == 0)
            B = 1;
    }
    const uint64_t bucket_size = (N + B - 1) / B;

    auto gid_to_bucket = [&](uint32_t gid) -> uint32_t {
        return std::min<uint32_t>((uint32_t)(gid / bucket_size), B - 1);
    };
    auto bucket_base = [&](uint32_t b) -> uint32_t {
        uint64_t base = (uint64_t)b * bucket_size;
        return (uint32_t)std::min<uint64_t>(base, N);
    };
    auto bucket_limit = [&](uint32_t b) -> uint32_t {
        uint64_t lim = std::min<uint64_t>(((uint64_t)b + 1) * bucket_size, N);
        return (uint32_t)lim;
    };

    if (prof_basic())
    {
        std::cout << "Buckets: B=" << B << " bucket_size≈" << bucket_size
                  << " (per_gid_bytes_inmem=" << per_gid_bytes_inmem << "B, reserve=" << reserve_bytes / 1024 / 1024
                  << "MiB)\n";
    }

    // ---------- Open Bucket Logs ----------
    BucketFDs bucket_fds;
    if (does_self_dump || does_cross_pairs)
    {
        bucket_fds.open_all(a.tag_prefix, B, /*truncate=*/does_self_dump);
#pragma omp parallel
        {
            tls_stage.init(1u << 18);
        } // staging per thread (256 KiB)
    }

    auto flush_bucket_buffers = [&]() {
        if (!(does_self_dump || does_cross_pairs))
            return;
#pragma omp parallel
        {
            if (tls_stage.used)
            {
                tls_stage.flush();
            }
        }
    };
    auto flush_bucket_logs = [&]() {
        if (!(does_self_dump || does_cross_pairs))
            return;
        flush_bucket_buffers();
        bucket_fds.close_all();
    };

    // ---------- Self-Subgraph Dump ----------
    double self_dump_s = 0.0;
    if (does_self_dump)
    {
        auto self_t0 = Clock::now();
        for (uint32_t t = 0; t < a.P; t++)
        {
            uint64_t ts = off[t], te = (t + 1 < a.P) ? off[t + 1] : N, tsize = te - ts;
            std::vector<std::vector<uint32_t>> adj;
            uint64_t md = 0, ec = 0;
            if (!load_graph_to_adj_memindex(subindex_prefix(a.out_prefix, t), tsize, adj, md, ec))
                die("load adj failed");
#pragma omp parallel for schedule(static)
            for (ptrdiff_t li = 0; li < (ptrdiff_t)tsize; li++)
            {
                uint32_t gid = (uint32_t)(ts + (uint64_t)li);
                auto &nb = adj[(size_t)li];
                if (nb.empty())
                    continue;
                std::vector<uint32_t> out;
                out.reserve(nb.size());
                for (uint32_t k = 0; k < nb.size(); k++)
                {
                    uint32_t v = (uint32_t)(ts + (uint64_t)nb[k]);
                    if (v == gid)
                        continue;
                    out.push_back(v);
                }
                if (!out.empty())
                {
                    uint32_t b = gid_to_bucket(gid);
                    int fd = bucket_fds.fds[b];
                    tls_stage.append_record(fd, (int)b, gid, out.data(), (uint16_t)out.size());
                }
            }
        }
        auto self_t1 = Clock::now();
        self_dump_s += sec(self_t0, self_t1);
    }

    if (a.stage == Args::Stage::SELF_DUMP)
    {
        flush_bucket_logs();
        if (prof_basic())
        {
            std::cout << "\n========== Timing ==========\n";
            std::cout << "Sub-index build calculation (s): " << build_calculation_s << "\n";
            std::cout << "Sub-index build total (s): " << build_s << "\n";
            std::cout << "Self-subgraph dump total (s): " << self_dump_s << "\n";
            std::cout << "\n========== I/O Summary ==========\n";
            std::cout << "Subindex total bytes (files): " << subindex_total_bytes << "\n";
        }
        return 0;
    }

    // ---------- Cross-Partition Search & Prune ----------
    double pair_s = 0.0;
    auto pair_t0 = Clock::now();

    double pair_total_s = 0.0;
    // global aggregates (no atomics; accumulated after each pair)
    double g_search_us = 0.0, g_prune_us = 0.0, g_write_us = 0.0;
    uint64_t g_queries = 0, g_visited = 0, g_pruned = 0;
    uint64_t g_write_ids = 0, g_write_recs = 0;

    std::unique_ptr<diskann::AbstractIndex> cached_next_u = nullptr;
    CrossPairMask cross_mask;
    const uint64_t cross_total_pairs = unordered_pair_count(a.P);
    uint64_t cross_pairs_considered = 0;
    uint64_t cross_pairs_processed = 0;

    if (does_cross_pairs)
    {
        cross_mask = load_cross_pair_mask(a);

        uint32_t anchor_begin = 0;
        uint32_t anchor_end_exclusive = (a.P > 1) ? (a.P - 1) : 0;
        if (a.stage == Args::Stage::CROSS_PAIRS)
        {
            if (a.P < 2)
            {
                anchor_begin = 0;
                anchor_end_exclusive = 0;
            }
            else if (a.anchor_start < 0 && a.anchor_end < 0)
            {
                anchor_begin = 0;
                anchor_end_exclusive = a.P - 1;
            }
            else
            {
                int64_t start = (a.anchor_start >= 0) ? a.anchor_start : 0;
                int64_t end = (a.anchor_end >= 0) ? a.anchor_end : start;
                if (start < 0 || end < start || end >= (int64_t)a.P - 1)
                    die("invalid cross-pairs anchor range");
                anchor_begin = (uint32_t)start;
                anchor_end_exclusive = (uint32_t)end + 1;
            }

            if (a.target >= 0)
            {
                if (a.anchor_start < 0 || a.anchor_end < 0 || a.anchor_start != a.anchor_end)
                    die("--target requires a single --anchor");
                if (a.target <= (int64_t)a.anchor_start || a.target >= (int64_t)a.P)
                    die("--target must satisfy anchor < target < P");
            }

            if (prof_basic())
            {
                std::cout << "[our_construction] Cross-pairs anchors: [" << anchor_begin << ", "
                          << anchor_end_exclusive << ")\n";
                if (a.target >= 0)
                    std::cout << "[our_construction] Cross-pairs target: " << a.target << "\n";
            }
        }
        if (prof_basic())
        {
            std::cout << "[our_construction] Cross-pair selection: "
                      << (cross_mask.enabled ? a.cross_pairs_file : "ALL")
                      << " selected_unordered_pairs="
                      << (cross_mask.enabled ? cross_mask.selected_pairs : cross_total_pairs)
                      << " total_unordered_pairs=" << cross_total_pairs << "\n";
        }

    for (uint32_t i = anchor_begin; i < anchor_end_exclusive; i++)
    {
        uint64_t is = off[i], ie = off[i + 1], isize = ie - is;

        std::unique_ptr<diskann::AbstractIndex> u_i;

        if (cached_next_u)
        {
            u_i = std::move(cached_next_u);
        }
        else
        {
            auto cfg_i = diskann::IndexConfigBuilder()
                             .with_metric(diskann::Metric::L2)
                             .with_dimension((size_t)D)
                             .with_max_points((size_t)isize)
                             .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                             .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                             .with_data_type(std::string("float"))
                             .is_dynamic_index(false)
                             .build();
            auto f_i = diskann::IndexFactory(cfg_i);
            u_i = f_i.create_instance();
            std::string i_prefix = subindex_prefix(a.out_prefix, i);
            u_i->load(i_prefix.c_str(), omp_get_max_threads(), a.L);
            drop_index_file_cache_if_enabled(i_prefix);
        }

        // Reverse order for j to enable index reuse (Paper Logic)
        for (uint32_t j = a.P - 1; j > i; j--)
        {
            if (a.target >= 0 && j != (uint32_t)a.target)
                continue;
            ++cross_pairs_considered;
            if (!cross_pair_selected(cross_mask, a.P, i, j))
                continue;
            ++cross_pairs_processed;

            uint64_t js = off[j], je = (j + 1 < a.P) ? off[j + 1] : N, jsize = je - js;

            auto cfg_j = diskann::IndexConfigBuilder()
                             .with_metric(diskann::Metric::L2)
                             .with_dimension((size_t)D)
                             .with_max_points((size_t)jsize)
                             .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                             .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                             .with_data_type(std::string("float"))
                             .is_dynamic_index(false)
                             .build();
            auto f_j = diskann::IndexFactory(cfg_j);
            auto u_j = f_j.create_instance();
            std::string j_prefix = subindex_prefix(a.out_prefix, j);
            u_j->load(j_prefix.c_str(), omp_get_max_threads(), a.L);
            drop_index_file_cache_if_enabled(j_prefix);

            auto pair_w0 = Clock::now();
#if defined(STATS)
            // per-pair aggregates
            double p_search_us = 0.0, p_prune_us = 0.0, p_write_us = 0.0;
            uint64_t p_queries = 0, p_visited = 0, p_pruned = 0;
            uint64_t p_write_ids = 0, p_write_recs = 0;
#endif

#pragma omp parallel
            {
#if defined(STATS)
                // (A) per-thread local accumulators (no per-query atomics)
                uint64_t l_search_us = 0, l_prune_us = 0, l_write_us = 0;
                uint64_t l_visited = 0, l_pruned = 0, l_queries = 0;
                uint64_t l_write_ids = 0, l_write_recs = 0;
#endif
                std::vector<uint32_t> visited_ids, pruned;
                std::vector<float> visited_dists;

                std::vector<uint32_t> self_ids;
                std::vector<float> self_d2;

                visited_ids.reserve(a.L * 2);
                visited_dists.reserve(a.L * 2);
                self_ids.reserve(a.R);
                self_d2.reserve(a.R);

                PairAccessorPtr acc{(diskann::Index<float> *)u_i.get(),
                                    (uint32_t)is,
                                    (uint32_t)ie,
                                    (diskann::Index<float> *)u_j.get(),
                                    (uint32_t)js,
                                    (uint32_t)je,
                                    (uint32_t)D};

#pragma omp for schedule(static)
                for (ptrdiff_t li = 0; li < (ptrdiff_t)isize; li++)
                {
                    self_ids.clear();
                    self_d2.clear();

                    uint32_t gid = (uint32_t)(is + (uint64_t)li);
                    const float *q = acc.idx_i->data_ptr_local((uint32_t)li);
#if defined(STATS)
                    uint64_t tS0 = thread_cpu_ns();
#endif
                    ((diskann::Index<float> *)u_j.get())
                        ->iterate_and_export_visited(q, a.L, visited_ids, &visited_dists, true);
#if defined(STATS)
                    uint64_t tS1 = thread_cpu_ns();
                    l_search_us += (tS1 - tS0) / 1000ull;
#endif

                    acc.load_self_neighbors_and_d2((uint32_t)li, q, self_ids, self_d2);
#if defined(STATS)
                    uint64_t tP0 = thread_cpu_ns();
#endif
                    pruned.clear();
                    pruned.reserve(a.R);

                    const size_t n1 = visited_ids.size();
                    const size_t n2 = self_ids.size();
                    const size_t n = n1 + n2;

                    if (n)
                    {
                        tls_arena.ensure(n * (sizeof(CN) + sizeof(float)), 64);
                        tls_arena.reset();

                        CN *pool = tls_arena.alloc_n<CN>(n);
                        size_t m = 0;

                        const uint32_t js32 = (uint32_t)js;
                        for (size_t k = 0; k < n1; ++k)
                        {
                            uint32_t v = js32 + visited_ids[k];
                            if (v == gid)
                                continue;
                            pool[m++] = CN{v, visited_dists[k]};
                        }
                        for (size_t k = 0; k < n2; ++k)
                        {
                            uint32_t v = self_ids[k];
                            if (v == gid)
                                continue;
                            pool[m++] = CN{v, self_d2[k]};
                        }

                        if (m)
                        {
                            occlude_list_ptr(gid, pool, m, /*alpha=*/1.0f, /*degree=*/a.R, pruned, acc);
                        }
                    }
#if defined(STATS)
                    uint64_t tP1 = thread_cpu_ns();
                    l_prune_us += (tP1 - tP0) / 1000ull;

                    uint64_t tW0 = thread_cpu_ns();
#endif
                    if (!pruned.empty())
                    {
                        size_t wcnt = 0;
                        for (size_t t = 0; t < pruned.size(); ++t)
                        {
                            uint32_t v = pruned[t];
                            if ((uint64_t)v >= js && (uint64_t)v < je)
                            {
                                pruned[wcnt++] = v;
                            }
                        }
                        if (wcnt)
                        {
                            uint32_t b = gid_to_bucket(gid);
                            int fd = bucket_fds.fds[b];
                            tls_stage.append_record(fd, (int)b, gid, pruned.data(), (uint16_t)wcnt);
#if defined(STATS)
                            l_write_recs += 1;
                            l_write_ids += wcnt;
#endif
                        }
                    }
#if defined(STATS)
                    uint64_t tW1 = thread_cpu_ns();
                    l_write_us += (tW1 - tW0) / 1000ull;

                    l_visited += (uint64_t)visited_ids.size();
                    l_pruned += (uint64_t)pruned.size();
                    l_queries += 1;
#endif
                }

#pragma omp for schedule(static)
                for (ptrdiff_t lj = 0; lj < (ptrdiff_t)jsize; lj++)
                {
                    self_ids.clear();
                    self_d2.clear();

                    uint32_t gid = (uint32_t)(js + (uint64_t)lj);
                    const float *q = acc.idx_j->data_ptr_local((uint32_t)lj);
#if defined(STATS)
                    uint64_t tS0 = thread_cpu_ns();
#endif
                    ((diskann::Index<float> *)u_i.get())
                        ->iterate_and_export_visited(q, a.L, visited_ids, &visited_dists, true);
#if defined(STATS)
                    uint64_t tS1 = thread_cpu_ns();
                    l_search_us += (tS1 - tS0) / 1000ull;
#endif

                    acc.load_self_neighbors_and_d2_j((uint32_t)lj, q, self_ids, self_d2);
#if defined(STATS)
                    uint64_t tP0 = thread_cpu_ns();
#endif
                    pruned.clear();
                    pruned.reserve(a.R);

                    const size_t n1 = visited_ids.size();
                    const size_t n2 = self_ids.size();
                    const size_t n = n1 + n2;

                    if (n)
                    {
                        tls_arena.ensure(n * (sizeof(CN) + sizeof(float)), 64);
                        tls_arena.reset();

                        CN *pool = tls_arena.alloc_n<CN>(n);
                        size_t m = 0;

                        const uint32_t is32 = (uint32_t)is;
                        for (size_t k = 0; k < n1; ++k)
                        {
                            uint32_t v = is32 + visited_ids[k];
                            if (v == gid)
                                continue;
                            pool[m++] = CN{v, visited_dists[k]};
                        }
                        for (size_t k = 0; k < n2; ++k)
                        {
                            uint32_t v = self_ids[k];
                            if (v == gid)
                                continue;
                            pool[m++] = CN{v, self_d2[k]};
                        }

                        if (m)
                        {
                            occlude_list_ptr(gid, pool, m, /*alpha=*/1.0f, /*degree=*/a.R, pruned, acc);
                        }
                    }
#if defined(STATS)
                    uint64_t tP1 = thread_cpu_ns();
                    l_prune_us += (tP1 - tP0) / 1000ull;

                    uint64_t tW0 = thread_cpu_ns();
#endif
                    if (!pruned.empty())
                    {
                        size_t wcnt = 0;
                        for (size_t t = 0; t < pruned.size(); ++t)
                        {
                            uint32_t v = pruned[t];
                            if ((uint64_t)v >= is && (uint64_t)v < ie)
                            {
                                pruned[wcnt++] = v;
                            }
                        }
                        if (wcnt)
                        {
                            uint32_t b = gid_to_bucket(gid);
                            int fd = bucket_fds.fds[b];
                            tls_stage.append_record(fd, (int)b, gid, pruned.data(), (uint16_t)wcnt);
#if defined(STATS)
                            l_write_recs += 1;
                            l_write_ids += wcnt;
#endif
                        }
                    }
#if defined(STATS)
                    uint64_t tW1 = thread_cpu_ns();
                    l_write_us += (tW1 - tW0) / 1000ull;

                    l_visited += (uint64_t)visited_ids.size();
                    l_pruned += (uint64_t)pruned.size();
                    l_queries += 1;
#endif
                }

#if defined(STATS)
// merge thread-local into pair-level (no atomics in hot path)
#pragma omp critical
                {
                    p_search_us += (double)l_search_us;
                    p_prune_us += (double)l_prune_us;
                    p_write_us += (double)l_write_us;
                    p_queries += l_queries;
                    p_visited += l_visited;
                    p_pruned += l_pruned;
                    p_write_ids += l_write_ids;
                    p_write_recs += l_write_recs;
                }
#endif
            } // parallel

            auto pair_w1 = Clock::now();
            double pair_wall = sec(pair_w0, pair_w1);
            pair_total_s += pair_wall;
#if defined(STATS)

            // pair-level stats print
            if (prof_basic() && p_queries > 0)
            {
                double as = p_search_us / p_queries;
                double ap = p_prune_us / p_queries;
                double aw = p_write_us / p_queries;
                double av = (double)p_visited / p_queries;
                double ar = (double)p_pruned / p_queries;

                double avg_ids_per_query = p_queries ? (double)p_write_ids / p_queries : 0.0;
                double frac_queries_wrote = p_queries ? (double)p_write_recs / p_queries : 0.0;
                double ids_per_written_rec = p_write_recs ? (double)p_write_ids / p_write_recs : 0.0;
                std::cout << "[pair] (i,j)=(" << i << "," << j << ")"
                          << " time=" << pair_wall << " s"
                          << " | avg_search_us=" << as << " | avg_prune_us=" << ap << " | avg_write_us=" << aw
                          << " | avg_visited=" << av << " | avg_pruned=" << ar
                          << " | avg_written_ids=" << avg_ids_per_query
                          << " | frac_queries_wrote=" << frac_queries_wrote
                          << " | ids_per_written_rec=" << ids_per_written_rec << " | n=" << p_queries << "\n";
            }

            // accumulate to global (outside parallel)
            g_search_us += p_search_us;
            g_prune_us += p_prune_us;
            g_write_us += p_write_us;
            g_queries += p_queries;
            g_visited += p_visited;
            g_pruned += p_pruned;
            g_write_ids += p_write_ids;
            g_write_recs += p_write_recs;
#endif
            // Logic from paper: Index reuse. If the target will not become the
            // next source, release it before the next target load so cgroup RSS
            // does not retain freed heap across sparse pairs.
            if (a.target >= 0)
            {
                u_j.reset();
                trim_released_heap();
            }
            else if (j == i + 1)
            {
                cached_next_u = std::move(u_j);
            }
            else
            {
                u_j.reset();
                trim_released_heap();
            }
        }
    }

        flush_bucket_logs();
        auto pair_t1 = Clock::now();
        pair_s += sec(pair_t0, pair_t1);
    }

    if (a.stage == Args::Stage::CROSS_PAIRS)
    {
        if (prof_basic())
        {
            std::cout << "\n========== Timing ==========\n";
            std::cout << "Cros-partition calculation (s): " << pair_total_s << "\n";
            std::cout << "Cross-partition total (s): " << pair_s << "\n";
            std::cout << "Cross-pair total unordered pairs: " << cross_total_pairs << "\n";
            std::cout << "Cross-pair selected unordered pairs: "
                      << (cross_mask.enabled ? cross_mask.selected_pairs : cross_total_pairs) << "\n";
            std::cout << "Cross-pair considered pairs in this run: " << cross_pairs_considered << "\n";
            std::cout << "Cross-pair processed pairs in this run: " << cross_pairs_processed << "\n";
        }
        return 0;
    }

    if (!does_compact)
        return 0;

    // ---------- Compaction Phase ----------
    auto c0 = Clock::now();

    const std::string idx_path = a.tag_prefix + "cand.idx";
    const std::string ids_path = a.tag_prefix + "cand.ids";
    int idx_fd = ::open(idx_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (idx_fd < 0)
        die("open cand.idx");
    int ids_fd = ::open(ids_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (ids_fd < 0)
        die("open cand.ids");

    struct IdxHdr
    {
        uint32_t magic, version;
        uint64_t N;
        uint64_t rsv[2];
    } idxhdr{0x43494458u, 1u, N, {0, 0}};
    if (::pwrite(idx_fd, &idxhdr, sizeof(idxhdr), 0) != (ssize_t)sizeof(idxhdr))
        die("write idx hdr");

    struct IdsHdr
    {
        uint32_t magic, version;
        uint64_t total_ids;
        uint64_t rsv[2];
    } idhdr{0x43494553u, 1u, 0ull, {0, 0}}; // placeholder, will rewrite later
    if (::pwrite(ids_fd, &idhdr, sizeof(idhdr), 0) != (ssize_t)sizeof(idhdr))
        die("write ids hdr");

    const uint64_t idx_body_off = 32ull, ids_body_off = 32ull;

    uint64_t acc_ids = 0; // total ids so far (elements)
    uint64_t comp_read_bytes = 0, comp_write_ids_bytes = 0;

    for (uint32_t b = 0; b < B; b++)
    {
        const uint32_t base = bucket_base(b), lim = bucket_limit(b);
        if (base >= lim)
            continue;

        const std::string p = bucket_log_name(a.tag_prefix, b);
        // stat size
        uint64_t fsz = 0;
        {
            struct stat st
            {
            };
            if (::stat(p.c_str(), &st) == 0)
                fsz = (uint64_t)st.st_size;
        }
        // load into memory once (disk 1-pass)
        std::vector<uint8_t> blob;
        blob.resize((size_t)fsz);
        if (fsz > 0)
        {
            int fd = ::open(p.c_str(), O_RDONLY);
            if (fd < 0)
                die("open bucket");
            if (construction_fadvise_enabled())
                posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

            const size_t CHUNK = std::max<size_t>((size_t)a.read_buf_mib << 20, 1ull << 20); // 1~R MiB
            size_t offb = 0;
            while (offb < (size_t)fsz)
            {
                size_t want = std::min(CHUNK, (size_t)fsz - offb);
                ssize_t n = ::read(fd, blob.data() + offb, want);
                if (n < 0)
                    die("read bucket");
                if (n == 0)
                    break;
                offb += (size_t)n;
            }
            if (construction_fadvise_enabled())
                posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
            comp_read_bytes += fsz;
        }

        const size_t U = (size_t)(lim - base);
        std::vector<uint32_t> counts(U, 0);
        uint64_t sum_b = 0;

        // Pass #1 (in-memory): counts/prefix + write cand.idx rows
        {
            size_t offm = 0, tot = (size_t)fsz;
            while (offm + sizeof(RecHdr) <= tot)
            {
                RecHdr rh{};
                std::memcpy(&rh, blob.data() + offm, sizeof(rh));
                if (rh.size_bytes < REC_HDR_ALIGNED)
                    break;
                if (offm + rh.size_bytes > tot)
                    break;
                if (rh.gid >= base && rh.gid < lim)
                {
                    size_t u = (size_t)(rh.gid - base);
                    counts[u] += rh.len;
                    sum_b += rh.len;
                }
                offm += rh.size_bytes;
            }

            // prefix and write idx rows
            std::vector<uint64_t> prefix(U + 1, 0);
            for (size_t u = 0; u < U; u++)
                prefix[u + 1] = prefix[u] + counts[u];

            for (uint32_t u = 0; u < (uint32_t)(lim - base); u++)
            {
                uint64_t off_elems = acc_ids + prefix[u];
                uint64_t off_byt = off_elems * 4ull;
                uint32_t cnt = counts[u];
                uint16_t fl = 0, pad = 0;
                off_t pos = (off_t)(idx_body_off + ((uint64_t)(base + u) * 16ull));
                struct
                {
                    uint64_t offB;
                    uint32_t cnt;
                    uint16_t fl;
                    uint16_t pd;
                } row{off_byt, cnt, fl, pad};
                if (::pwrite(idx_fd, &row, 16, pos) != 16)
                    die("write idx row");
            }

            // Pass #2 (in-memory): fill out_ids then single pwrite to cand.ids
            if (sum_b > 0)
            {
                std::vector<uint32_t> out_ids((size_t)sum_b);
                std::vector<uint64_t> cursor = prefix;

                size_t offm2 = 0;
                while (offm2 + sizeof(RecHdr) <= (size_t)fsz)
                {
                    RecHdr rh{};
                    std::memcpy(&rh, blob.data() + offm2, sizeof(rh));
                    if (rh.size_bytes < REC_HDR_ALIGNED)
                        break;
                    if (offm2 + rh.size_bytes > (size_t)fsz)
                        break;
                    const uint8_t *rec = blob.data() + offm2 + sizeof(rh);
                    if (rh.len && rh.gid >= base && rh.gid < lim)
                    {
                        size_t u = (size_t)(rh.gid - base);
                        uint64_t oe = cursor[u];
                        std::memcpy(out_ids.data() + oe, rec, (size_t)rh.len * 4u);
                        cursor[u] += rh.len;
                    }
                    offm2 += rh.size_bytes;
                }

                const uint64_t off_bytes = acc_ids * 4ull;
                ssize_t wn = ::pwrite(ids_fd, out_ids.data(), (size_t)sum_b * 4u, (off_t)(ids_body_off + off_bytes));
                if (wn != (ssize_t)((size_t)sum_b * 4u))
                    die("pwrite cand.ids bucket");
                comp_write_ids_bytes += (uint64_t)((size_t)sum_b * 4u);

                // Flush & advise (minimize cache residency)
                if (construction_sync_before_drop_enabled())
                    ::fdatasync(ids_fd);
                if (construction_fadvise_enabled())
                    ::posix_fadvise(ids_fd, (off_t)(ids_body_off + off_bytes), (off_t)((size_t)sum_b * 4u),
                                    POSIX_FADV_DONTNEED);
            }
        }

        // delete bucket log
        if (fsz > 0)
            std::remove(p.c_str());
        acc_ids += sum_b;
    }

    // finalize cand.ids header
    {
        struct IdsHdr
        {
            uint32_t magic, version;
            uint64_t total_ids;
            uint64_t rsv[2];
        } final_idhdr{0x43494553u, 1u, acc_ids, {0, 0}};
        if (::pwrite(ids_fd, &final_idhdr, sizeof(final_idhdr), 0) != (ssize_t)sizeof(final_idhdr))
            die("rewrite ids hdr");
    }

    ::close(idx_fd);
    ::close(ids_fd);

    auto c1 = Clock::now();

    // ---------- Optionally delete subindex files ----------
    if (!a.keep_subindex)
    {
        for (uint32_t t = 0; t < a.P; t++)
        {
            std::string t_pref = subindex_prefix(a.out_prefix, t);
            std::remove(t_pref.c_str());
            std::remove((t_pref + ".data").c_str());
        }
    }

    // ---------- Final timing & I/O summary ----------
    if (prof_basic())
    {
        std::cout << "\n========== Timing ==========\n";
        if (a.stage != Args::Stage::COMPACT)
        {
            std::cout << "Sub-index build calculation (s): " << build_calculation_s << "\n";
            std::cout << "Sub-index build total (s): " << build_s << "\n";
            std::cout << "Self-subgraph dump total (s): " << self_dump_s << "\n";
            std::cout << "Cros-partition calculation (s): " << pair_total_s << "\n";
            std::cout << "Cross-partition total (s): " << pair_s << "\n";
        }
        std::cout << "Compact (s): " << sec(c0, c1) << "\n";

        if (a.stage != Args::Stage::COMPACT && g_queries > 0)
        {
            double as = g_search_us / g_queries;
            double ap = g_prune_us / g_queries;
            double aw = g_write_us / g_queries;
            double av = (double)g_visited / g_queries;
            double ar = (double)g_pruned / g_queries;

            double avg_ids_per_query = (double)g_write_ids / g_queries;
            double frac_queries_wrote = (double)g_write_recs / g_queries;
            double ids_per_written_rec = g_write_recs ? (double)g_write_ids / g_write_recs : 0.0;
            std::cout << "== Pair Summary (all) == "
                      << "avg_search_us=" << as << " | "
                      << "avg_prune_us=" << ap << " | "
                      << "avg_write_us=" << aw << " | "
                      << "avg_visited=" << av << " | "
                      << "avg_pruned=" << ar << " | "
                      << "avg_written_ids=" << avg_ids_per_query << " | "
                      << "frac_queries_wrote=" << frac_queries_wrote << " | "
                      << "ids_per_written_rec=" << ids_per_written_rec << " | "
                      << "n=" << g_queries << "\n";
        }

        uint64_t cand_ids_total = acc_ids * 4ull + 32ull;
        uint64_t cand_idx_total = 32ull + N * 16ull;

        std::cout << "\n========== I/O Summary ==========\n";
        std::cout << "Subindex total bytes (files): " << subindex_total_bytes << "\n";
        std::cout << "Compaction read (bucket logs): " << comp_read_bytes << " bytes\n";
        std::cout << "Compaction write cand.ids: " << comp_write_ids_bytes << " bytes (total file ~" << cand_ids_total
                  << ")\n";
        std::cout << "Compaction write cand.idx: " << (N * 16ull) << " bytes (total file ~" << cand_idx_total << ")\n";
        std::cout << "Output: " << (idx_path) << " , " << (ids_path) << "\n";
        if (!a.keep_subindex)
            std::cout << "Subindex files removed.\n";
    }

    return 0;
}

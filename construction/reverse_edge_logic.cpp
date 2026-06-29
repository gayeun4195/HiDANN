/*
 * reverse_edge_logic.cpp
 *
 * Purpose:
 *   Builds a symmetric candidate graph from an input directed graph.
 *   Adds reverse edges (if u -> v, ensure v -> u) and saves the result
 *   as a candidate graph (cand.idx/cand.ids).
 *   Used to densify the graph before pruning, improving recall.
 *
 * Usage:
 *   ./reverse_edge_logic --graph=graph.mem.index --data=base.fbin --out-prefix=PREFIX ...
 */

#include <algorithm>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
static constexpr uint64_t READ_DROP_INTERVAL = 4ull * 1024ull * 1024ull;
static constexpr size_t WRITE_DROP_INTERVAL = 64ull * 1024ull * 1024ull;

// ---------- Utility Functions ----------
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
static inline void _read_exact(std::ifstream &in, void *dst, size_t n)
{
    in.read(reinterpret_cast<char *>(dst), (std::streamsize)n);
    if (!in)
        die("short read");
}
static inline bool path_exists(const std::string &p)
{
    std::ifstream f(p, std::ios::binary);
    return (bool)f;
}
static inline uint64_t file_size(const std::string &p)
{
    struct stat st
    {
    };
    if (::stat(p.c_str(), &st) != 0)
        die("stat failed: " + p);
    return (uint64_t)st.st_size;
}

static inline void fadvise_dontneed(int fd, uint64_t off, uint64_t len, const char *label)
{
    if (fd < 0 || !construction_fadvise_enabled())
        return;
    int ret = ::posix_fadvise(fd, (off_t)off, (off_t)len, POSIX_FADV_DONTNEED);
    if (ret != 0)
    {
        std::cerr << "warning: posix_fadvise(DONTNEED) failed for " << label << ": " << std::strerror(ret) << "\n";
    }
}

static inline void drop_file_cache(const std::string &path)
{
    if (!construction_fadvise_enabled())
        return;
    int fd = ::open(path.c_str(), O_RDONLY | O_LARGEFILE);
    if (fd < 0)
        return;
    fadvise_dontneed(fd, 0, 0, path.c_str());
    ::close(fd);
}

static inline void flush_and_drop_range(int fd, uint64_t off, uint64_t len, const char *label)
{
    if (fd < 0 || len == 0)
        return;
    if (construction_sync_before_drop_enabled() && ::fdatasync(fd) != 0)
    {
        std::cerr << "warning: fdatasync failed for " << label << ": " << std::strerror(errno) << "\n";
    }
    fadvise_dontneed(fd, off, len, label);
}

class SequentialReadDropper
{
  public:
    explicit SequentialReadDropper(const std::string &path, uint64_t interval_bytes = READ_DROP_INTERVAL)
        : path_(path), interval_bytes_(interval_bytes)
    {
        fd_ = ::open(path.c_str(), O_RDONLY | O_LARGEFILE);
    }

    ~SequentialReadDropper()
    {
        if (fd_ >= 0)
            ::close(fd_);
    }

    void advance(uint64_t bytes)
    {
        pos_ += bytes;
        if (fd_ < 0 || !construction_fadvise_enabled() || interval_bytes_ == 0 ||
            pos_ - last_drop_ < interval_bytes_)
            return;
        int ret = ::posix_fadvise(fd_, (off_t)last_drop_, (off_t)(pos_ - last_drop_), POSIX_FADV_DONTNEED);
        if (ret != 0 && !warned_)
        {
            std::cerr << "warning: posix_fadvise(DONTNEED) failed for " << path_ << ": " << std::strerror(ret) << "\n";
            warned_ = true;
        }
        last_drop_ = pos_;
    }

  private:
    std::string path_;
    int fd_ = -1;
    uint64_t interval_bytes_ = 0;
    uint64_t pos_ = 0;
    uint64_t last_drop_ = 0;
    bool warned_ = false;
};

// Matches apply_pq_prune header
#pragma pack(push, 1)
struct GraphHdr
{
    uint64_t isz;    // final file size (bytes)
    uint32_t maxObs; // max observed degree
    uint32_t ep;     // unused
    uint64_t nf;     // unused
};

// ---------- Candidate File Layouts ----------
struct CandIdxHdr
{
    uint32_t magic, version; // magic=0x43494458 ('C''I''D''X'), version=1
    uint64_t N;
    uint64_t rsv[2];
};
struct CandIdxRow
{
    uint64_t off_bytes; // offset from start of ids stream (after CandIdsHdr)
    uint32_t cnt;
    uint16_t flags;
    uint16_t pad;
};
struct CandIdsHdr
{
    uint32_t magic, version; // magic=0x43494553 ('C''I''E''S'), version=1
    uint64_t total_ids;
    uint64_t rsv[2];
};
#pragma pack(pop)

static constexpr long double STRIPE_PAYLOAD_FILL = 0.75L;
static constexpr uint64_t DEFAULT_STRIPE_MAX_MIB = 1536;

// ---------- CLI Arguments ----------
struct Args
{
    std::string graph_path; // input graph.mem.index (deg | ids...)
    std::string data_path;  // base.fbin (to read D and cross-check N if desired)
    std::string out_prefix; // output prefix for cand.* (prefix + "cand.idx", prefix + "cand.ids")
    uint32_t R = 32;        // for mem budget formula
    uint32_t P = 0;         // partitions (required)
    float safety_frac = 0.05f;
    uint64_t max_mib = DEFAULT_STRIPE_MAX_MIB; // optional hard cap for stripe buffer (MiB). 0 -> disabled.
    uint64_t memory_budget_bytes = 0; // explicit logical build memory budget
    int verbose = 1;
};

static void parse_flags(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; i++)
    {
        const char *s = argv[i];
        auto pref = [&](const char *k) {
            return std::strncmp(s, k, std::strlen(k)) == 0 ? (s + std::strlen(k)) : nullptr;
        };
        if (auto v = pref("--graph="))
            a.graph_path = v;
        else if (auto v = pref("--data="))
            a.data_path = v;
        else if (auto v = pref("--out-prefix="))
            a.out_prefix = v;
        else if (auto v = pref("--R="))
            a.R = std::stoul(v);
        else if (auto v = pref("--P="))
            a.P = std::stoul(v);
        else if (auto v = pref("--safety-frac="))
            a.safety_frac = std::stof(v);
        else if (auto v = pref("--max-mib="))
            a.max_mib = std::stoull(v);
        else if (auto v = pref("--memory-budget-bytes="))
            a.memory_budget_bytes = std::stoull(v);
        else if (std::strcmp(s, "--quiet") == 0)
            a.verbose = 0;
    }
}

static inline void read_exact_tracked(std::ifstream &in, void *dst, size_t n, SequentialReadDropper *dropper)
{
    _read_exact(in, dst, n);
    if (dropper)
        dropper->advance(n);
}

static inline uint32_t read_u32(std::ifstream &in, SequentialReadDropper *dropper = nullptr)
{
    uint32_t x = 0;
    read_exact_tracked(in, &x, 4, dropper);
    return x;
}
static inline void skip_bytes(std::ifstream &in, uint64_t bytes, SequentialReadDropper *dropper = nullptr)
{
    in.seekg((std::streamoff)bytes, std::ios::cur);
    if (!in)
        die("seek fail");
    if (dropper)
        dropper->advance(bytes);
}

// ---------- Main ----------
int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: ./build_symmetric_candidates "
                     "--graph=graph.mem.index --data=base.fbin --out-prefix=PREFIX "
                     "[--R=32] [--P=20] [--safety-frac=0.05] [--max-mib=1536] [--memory-budget-bytes=N]\n";
        return 1;
    }
    Args a;
    parse_flags(argc, argv, a);
    std::cout << "[reverse_edge_logic] HIDANN_CONSTRUCTION_FADVISE=" << (construction_fadvise_enabled() ? 1 : 0)
              << " HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=" << (construction_sync_before_drop_enabled() ? 1 : 0)
              << "\n";
    if (a.graph_path.empty() || a.data_path.empty() || a.out_prefix.empty())
        die("missing required flags");
    if (a.P == 0)
        die("please pass --P=<partitions>");

    auto prog_start = Clock::now();

    // ---------- Read Base Data Header ----------
    uint64_t N_data = 0, D_data = 0;
    {
        std::ifstream in(a.data_path, std::ios::binary);
        if (!in)
            die("open data");
        uint32_t n32 = 0, d32 = 0;
        _read_exact(in, &n32, 4);
        _read_exact(in, &d32, 4);
        N_data = n32;
        D_data = d32;
    }

    // ---------- Open Graph and Count N ----------
    (void)file_size; // keep symbol used if needed elsewhere
    std::ifstream gin(a.graph_path, std::ios::binary);
    if (!gin)
        die("open graph");
    GraphHdr gh{};
    _read_exact(gin, &gh, sizeof(gh));

    // Pass 0: count N
    SequentialReadDropper pass0_dropper(a.graph_path);
    pass0_dropper.advance(sizeof(GraphHdr));
    auto pass0_start = Clock::now();
    uint64_t N = 0;
    while (gin)
    {
        if (gin.peek() == std::char_traits<char>::eof())
            break;
        uint32_t deg = read_u32(gin, &pass0_dropper);
        skip_bytes(gin, (uint64_t)deg * 4ull, &pass0_dropper);
        ++N;
    }
    auto pass0_end = Clock::now();
    drop_file_cache(a.graph_path);

    if (N == 0)
        die("graph seems empty");
    if (N_data && N_data != N)
    {
        if (a.verbose)
        {
            std::cerr << "warning: base.fbin N(" << N_data << ") != graph nodes(" << N << ") — proceeding with N=" << N
                      << "\n";
        }
    }

    // ---- Memory budget (same formula as apply_pq_prune) ----
    long double per_node_ld = (long double)D_data * 4.0L + (long double)a.R * 4.0L;
    uint64_t formula_budget_bytes = (uint64_t)(2.0L * (long double)N * per_node_ld / (long double)a.P);
    uint64_t mem_budget_bytes = a.memory_budget_bytes > 0 ? a.memory_budget_bytes : formula_budget_bytes;
    uint64_t safety = (uint64_t)((long double)mem_budget_bytes * (long double)a.safety_frac);

    // ---------- Pass 1: Count Symmetric Candidates ----------
    gin.clear();
    gin.seekg((std::streamoff)sizeof(GraphHdr), std::ios::beg);
    if (!gin)
        die("seek graph to records");
    SequentialReadDropper pass1_dropper(a.graph_path);
    pass1_dropper.advance(sizeof(GraphHdr));

    std::vector<uint32_t> cnt(N, 0);
    uint64_t edges_seen = 0;

    auto pass1_start = Clock::now();
    for (uint64_t u = 0; u < N; ++u)
    {
        uint32_t deg = read_u32(gin, &pass1_dropper);
        // Forward contributions to u
        cnt[u] += deg;
        // Reverse contributions to neighbors
        for (uint32_t i = 0; i < deg; ++i)
        {
            uint32_t v = read_u32(gin, &pass1_dropper);
            if (v >= N)
                die("neighbor id out of range");
            cnt[v] += 1;
        }
        edges_seen += deg;
    }
    auto pass1_end = Clock::now();
    drop_file_cache(a.graph_path);

    // ---------- Compute Offsets ----------
    std::vector<uint64_t> off_elems(N + 1, 0);
    for (uint64_t i = 0; i < N; ++i)
        off_elems[i + 1] = off_elems[i] + (uint64_t)cnt[i];
    uint64_t total_ids = off_elems[N];

    // ---------- Write cand.idx (Index File) ----------
    const std::string idx_path = a.out_prefix + "cand.idx";
    const std::string ids_path = a.out_prefix + "cand.ids";

    int idx_fd = ::open(idx_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_LARGEFILE, 0644);
    if (idx_fd < 0)
        die("open cand.idx for write");
    {
        CandIdxHdr ihdr{0x43494458u, 1u, N, {0ull, 0ull}};
        if (::write(idx_fd, &ihdr, sizeof(ihdr)) != (ssize_t)sizeof(ihdr))
            die("write cand.idx hdr");
        // Emit cand.idx rows in blocks; avoid a resident rows[N] allocation.
        const uint64_t row_block_rows = 1ull << 20;
        std::vector<CandIdxRow> rows((size_t)std::min<uint64_t>(N, row_block_rows));
        for (uint64_t base = 0; base < N; base += row_block_rows)
        {
            const uint64_t rows_this = std::min<uint64_t>(row_block_rows, N - base);
            for (uint64_t i = 0; i < rows_this; ++i)
            {
                const uint64_t u = base + i;
                rows[(size_t)i].off_bytes = off_elems[u] * 4ull; // from after CandIdsHdr
                rows[(size_t)i].cnt = cnt[u];
                rows[(size_t)i].flags = 0;
                rows[(size_t)i].pad = 0;
            }

            size_t bytes = (size_t)rows_this * sizeof(CandIdxRow);
            size_t done = 0;
            const uint8_t *p = reinterpret_cast<const uint8_t *>(rows.data());
            while (done < bytes)
            {
                ssize_t wn = ::write(idx_fd, p + done, bytes - done);
                if (wn < 0)
                    die("write cand.idx rows");
                done += (size_t)wn;
            }
        }
        if (construction_sync_before_drop_enabled())
            ::fsync(idx_fd);
        fadvise_dontneed(idx_fd, 0, 0, idx_path.c_str());
        ::close(idx_fd);
    }

    // ---------- Prepare cand.ids (Data File) ----------
    int ids_fd = ::open(ids_path.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_LARGEFILE, 0644);
    if (ids_fd < 0)
        die("open cand.ids for write");
    {
        CandIdsHdr shdr{0x43494553u, 1u, total_ids, {0ull, 0ull}};
        if (::write(ids_fd, &shdr, sizeof(shdr)) != (ssize_t)sizeof(shdr))
            die("write cand.ids hdr");
        // Pre-allocate file space: punch zeros using lseek then single write of 1 byte
        if (total_ids)
        {
            off_t end_off = (off_t)(sizeof(CandIdsHdr) + total_ids * 4ull);
            if (::lseek(ids_fd, end_off - 1, SEEK_SET) == (off_t)-1)
                die("lseek ids prealloc");
            uint8_t z = 0;
            if (::write(ids_fd, &z, 1) != 1)
                die("prealloc write");
        }
        if (construction_sync_before_drop_enabled())
            ::fsync(ids_fd);
        fadvise_dontneed(ids_fd, 0, 0, ids_path.c_str());
    }

    // ---------- Memory Budget & Feasibility Check ----------
    auto ids_bytes_of = [&](uint64_t lo, uint64_t hi) -> uint64_t { return (off_elems[hi] - off_elems[lo]) * 4ull; };

    const uint64_t SMALL_SLACK = 4ull * 1024ull * 1024ull;              // 4 MiB
    auto static_overhead_bytes = (uint64_t)(4ull * N + 8ull * (N + 1)); // 4N + 8(N+1)
    auto dyn_overhead = [&](uint64_t U) -> uint64_t {
        return static_overhead_bytes + 8ull * U + SMALL_SLACK; // + 8U + slack
    };
    auto avail_for_U = [&](uint64_t U) -> uint64_t {
        // budget after safety minus overhead(U); also respect --max-mib cap if given
        if (U == 0)
            return 0;
        uint64_t avail = (mem_budget_bytes > safety) ? (mem_budget_bytes - safety) : 0ull;
        if (avail <= dyn_overhead(U))
            return 0ull;
        avail -= dyn_overhead(U);
        avail = (uint64_t)((long double)avail * STRIPE_PAYLOAD_FILL);
        if (avail == 0)
            return 0ull;
        if (a.max_mib)
        {
            uint64_t cap = a.max_mib * 1024ull * 1024ull;
            avail = std::min(avail, cap);
        }
        return avail;
    };

    // Feasibility check BEFORE filling: can we place at least U=1 for the first node?
    {
        uint64_t lo = 0, hi = 1;
        uint64_t need = ids_bytes_of(lo, hi);
        uint64_t avail = avail_for_U(1);
        if (avail == 0 || need > avail)
        {
            std::cerr << "fatal: insufficient memory for even a single-node stripe.\n"
                      << "  need(U=1)=" << (need / (1024.0 * 1024.0)) << " MiB"
                      << " | avail(U=1)=" << (avail / (1024.0 * 1024.0)) << " MiB\n"
                      << "  budget=" << (mem_budget_bytes / (1024.0 * 1024.0)) << " MiB"
                      << " | safety=" << (safety / (1024.0 * 1024.0)) << " MiB"
                      << " | static_overhead=" << (static_overhead_bytes / (1024.0 * 1024.0)) << " MiB"
                      << " | slack=" << (SMALL_SLACK / (1024.0 * 1024.0)) << " MiB\n";
            return 2;
        }
        if (a.verbose && avail < (64ull * 1024ull * 1024ull))
        {
            std::cerr << "warning: avail(U=1) < 64 MiB, performance may degrade\n";
        }
    }

    // pick_hi considers overhead(U) inside the inequality
    auto pick_hi = [&](uint64_t lo) -> uint64_t {
        uint64_t L = lo + 1, R = N, best = L;
        while (L <= R)
        {
            uint64_t mid = (L + R) >> 1;
            uint64_t U = mid - lo;
            uint64_t avail = avail_for_U(U);
            if (avail == 0)
            {
                R = mid - 1;
                continue;
            }
            uint64_t need = ids_bytes_of(lo, mid);
            if (need <= avail)
            {
                best = mid;
                L = mid + 1;
            }
            else
                R = mid - 1;
        }
        return best;
    };
    // ========= END dynamic overhead =========

    // ---------- Populating Reverse Edges (Stripe Batching) ----------
    uint64_t ids_data_off0 = sizeof(CandIdsHdr);

    auto fill_start = Clock::now();
    uint64_t stripes = 0;
    uint64_t maxU = 0;            // for overhead reporting
    double first_avail_mib = 0.0; // show avail for first stripe in final log
    std::vector<uint8_t> buf;
    std::vector<uint64_t> cur_bytes;
    std::vector<uint32_t> nbr;
    if (a.max_mib)
    {
        const uint64_t cap = a.max_mib * 1024ull * 1024ull;
        buf.reserve((size_t)cap);
    }
    for (uint64_t lo = 0; lo < N;)
    {
        uint64_t hi = pick_hi(lo);
        if (hi <= lo)
            hi = lo + 1; // ensure progress
        uint64_t U = hi - lo;
        uint64_t stripe_bytes = ids_bytes_of(lo, hi);
        if (stripe_bytes == 0)
        {
            lo = hi;
            continue;
        }

        uint64_t avail_now = avail_for_U(U);
        if (stripes == 0)
            first_avail_mib = avail_now / (1024.0 * 1024.0);
        if (a.verbose)
        {
            std::cerr << "[stripe] nodes=[" << lo << "," << hi << ") U=" << U
                      << " bytes=" << stripe_bytes / 1024.0 / 1024.0 << " MiB | avail=" << avail_now / 1024.0 / 1024.0
                      << " MiB\n";
        }

        // Allocate stripe buffer
        buf.resize((size_t)stripe_bytes);
        // Build per-node cursors (relative indices in 'buf' as bytes)
        cur_bytes.resize((size_t)U);
        for (uint64_t u = lo; u < hi; ++u)
        {
            cur_bytes[u - lo] = (off_elems[u] - off_elems[lo]) * 4ull;
        }

        // Scan the source graph (again)
        gin.clear();
        gin.seekg((std::streamoff)sizeof(GraphHdr), std::ios::beg);
        if (!gin)
            die("seek graph for stripe");
        SequentialReadDropper stripe_dropper(a.graph_path);
        stripe_dropper.advance(sizeof(GraphHdr));

        for (uint64_t u = 0; u < N; ++u)
        {
            uint32_t deg = read_u32(gin, &stripe_dropper);
            // Reuse the same neighbor buffer across nodes to reduce allocator churn.
            nbr.resize(deg);
            if (deg)
                read_exact_tracked(gin, nbr.data(), (size_t)deg * 4, &stripe_dropper);

            // Forward contributions: if u in stripe -> write its nbrs into u's segment
            if (u >= lo && u < hi)
            {
                uint64_t base = cur_bytes[u - lo];
                uint8_t *wp = buf.data() + base;
                size_t bytes = (size_t)deg * 4u;
                if (bytes)
                    std::memcpy(wp, nbr.data(), bytes);
                cur_bytes[u - lo] += bytes;
            }
            // Reverse contributions: for each v, if v in stripe -> write 'u' into v's segment
            for (uint32_t v : nbr)
            {
                if (v >= lo && v < hi)
                {
                    uint64_t pos = cur_bytes[v - lo];
                    if (pos + 4 > stripe_bytes)
                        die("cursor overflow");
                    std::memcpy(buf.data() + pos, &u, 4);
                    cur_bytes[v - lo] += 4;
                }
            }
        }
        drop_file_cache(a.graph_path);

        // Sanity: each cursor must reach its node-end
        for (uint64_t u = lo; u < hi; ++u)
        {
            uint64_t expect = (off_elems[u + 1] - off_elems[lo]) * 4ull;
            if (cur_bytes[u - lo] != expect)
            {
                std::cerr << "cursor mismatch at node " << u << " cur=" << cur_bytes[u - lo] << " expect=" << expect
                          << "\n";
                die("stripe fill mismatch");
            }
        }

        // Single pwrite for the stripe
        uint64_t global_off = ids_data_off0 + off_elems[lo] * 4ull;
        size_t done = 0, bytes = (size_t)stripe_bytes;
        uint64_t drop_start = global_off;
        while (done < bytes)
        {
            size_t want = std::min(WRITE_DROP_INTERVAL, bytes - done);
            ssize_t wn = ::pwrite(ids_fd, buf.data() + done, want, (off_t)(global_off + done));
            if (wn < 0)
                die("pwrite stripe cand.ids");
            done += (size_t)wn;
            uint64_t written_end = global_off + done;
            if (written_end - drop_start >= WRITE_DROP_INTERVAL)
            {
                flush_and_drop_range(ids_fd, drop_start, written_end - drop_start, ids_path.c_str());
                drop_start = written_end;
            }
        }
        flush_and_drop_range(ids_fd, drop_start, global_off + done - drop_start, ids_path.c_str());
        ++stripes;
        if (U > maxU)
            maxU = U;
        lo = hi;
    }
    auto fill_end = Clock::now();

    ::fsync(ids_fd);
    ::close(ids_fd);

    // ---------- Finalize and Report ----------
    double tN = sec(pass0_start, pass0_end);
    double tCount = sec(pass1_start, pass1_end);
    double tFill = sec(fill_start, fill_end);
    auto prog_end = Clock::now();
    double tTotal = sec(prog_start, prog_end);

    double totMiB = (double)(total_ids * 4ull) / (1024.0 * 1024.0);

    double static_over_mib = static_overhead_bytes / (1024.0 * 1024.0);
    double dyn_over_max_mib = (8.0 * maxU) / (1024.0 * 1024.0);
    double slack_mib = SMALL_SLACK / (1024.0 * 1024.0);

    std::cout << "symmetric cand generation done\n"
              << "  graph_nodes=" << N << " data(N,D)=(" << N_data << "," << D_data << ")\n"
              << "  edges=" << edges_seen << " (directed)\n"
              << "  total_ids=" << total_ids << " (" << totMiB << " MiB)\n"
              << "  mem_budget=" << (mem_budget_bytes / 1024.0 / 1024.0) << " MiB"
              << " | source=" << (a.memory_budget_bytes > 0 ? "explicit" : "P-derived")
              << " | p_derived=" << (formula_budget_bytes / 1024.0 / 1024.0) << " MiB"
              << " | safety=" << a.safety_frac
              << " | stripe_fill=" << (double)STRIPE_PAYLOAD_FILL
              << " | avail=" << first_avail_mib << " MiB\n"
              << "  stripes=" << stripes << "\n"
              << "  overhead_static=" << static_over_mib << " MiB"
              << " | overhead_dynamic_max=" << dyn_over_max_mib << " MiB"
              << " | slack=" << slack_mib << " MiB\n"
              << "  t_pass0(N)=" << tN << " s"
              << " | t_count=" << tCount << " s"
              << " | t_fill=" << tFill << " s\n"
              << "  total_runtime=" << tTotal << " s\n";

    return 0;
}

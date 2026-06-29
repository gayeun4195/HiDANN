/*
 * our_pruning.cpp
 *
 * Purpose:
 *   Performs graph pruning using Product Quantization (PQ) compressed distances.
 *   Faster but less accurate substitute for exact pruning.
 *   Generates PQ codes/pivots if missing, then prunes the candidate graph.
 *
 * OPTIMIZED VERSION (Streaming Metadata):
 *   - Does NOT load the entire cand.idx or auxiliary vectors (cnt, off, etc.) into memory.
 *   - Reads metadata in batches to keep memory footprint low (small % of available RAM).
 *   - Ensures strictly limited memory usage for high-dimensional or large datasets.
 *
 * Usage:
 *   ./our_pruning --data=base.fbin --cand-prefix=PREFIX --out=graph.mem.index ...
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
#include <omp.h>

#include "pq.h"        // diskann::generate_quantized_data
#include "pq_common.h" // NUM_PQ_CENTROIDS (=256)

// ---------- Utility Functions ----------

using Clock = std::chrono::steady_clock;
static constexpr long double PRUNE_BATCH_PAYLOAD_FILL = 0.50L;
static constexpr uint64_t PRUNE_BATCH_MAX_BYTES = 768ull * 1024ull * 1024ull;
static constexpr size_t WRITE_DROP_INTERVAL = 64ull * 1024ull * 1024ull;

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

// ---------- CLI Arguments ----------
struct Args
{
    std::string data_path;
    std::string cand_prefix;
    std::string out_graph;

    std::string pivots_path;
    std::string codes_path;

    uint32_t QD = 0; // PQ chunks (0 -> from pivots)
    uint32_t R = 32; // target out-degree
    uint32_t T = 1;  // T-occlusion threshold
    uint32_t P = 0;  // partitions (for mem budget formula)
    uint32_t threads = 0;
    float p_val = 0.05f;       // PQ training sampling fraction
    float safety_frac = 0.05f; // fraction of mem_budget reserved as headroom
    int reuse = 1;             // reuse existing PQ artifacts if present
    uint64_t chunk_bytes = 0;  // dynamic chunk budget from --chunk-bytes
    uint64_t memory_budget_bytes = 0; // explicit logical build memory budget
};
static void parse_flags(int argc, char **argv, Args &a)
{
    for (int i = 1; i < argc; i++)
    {
        const char *s = argv[i];
        auto pref = [&](const char *k) {
            return std::strncmp(s, k, std::strlen(k)) == 0 ? (s + std::strlen(k)) : nullptr;
        };
        if (auto v = pref("--data="))
            a.data_path = v;
        else if (auto v = pref("--cand-prefix="))
            a.cand_prefix = v;
        else if (auto v = pref("--out="))
            a.out_graph = v;
        else if (auto v = pref("--pivots="))
            a.pivots_path = v;
        else if (auto v = pref("--codes="))
            a.codes_path = v;
        else if (auto v = pref("--QD="))
            a.QD = std::stoul(v);
        else if (auto v = pref("--R="))
            a.R = std::stoul(v);
        else if (auto v = pref("--T="))
            a.T = std::stoul(v);
        else if (auto v = pref("--P="))
            a.P = std::stoul(v);
        else if (auto v = pref("--threads="))
            a.threads = std::stoul(v);
        else if (auto v = pref("--pval="))
            a.p_val = std::stof(v);
        else if (auto v = pref("--reuse="))
            a.reuse = std::atoi(v);
        else if (auto v = pref("--safety-frac="))
            a.safety_frac = std::stof(v);
        else if (auto v = pref("--chunk-bytes="))
            a.chunk_bytes = std::stoull(v);
        else if (auto v = pref("--memory-budget-bytes="))
            a.memory_budget_bytes = std::stoull(v);
    }
}

// ---------- Candidate File Layouts ----------
#pragma pack(push, 1)
struct CandIdxHdr
{
    uint32_t magic, version;
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
    uint32_t magic, version;
    uint64_t total_ids;
    uint64_t rsv[2];
};
#pragma pack(pop)

// ---------- PQ Artifacts and Logic ----------
static constexpr int NUM_CENTROIDS = 256;

struct PQSymL2
{
    std::vector<float> pivots;       // [256 * D], row-major
    std::vector<uint32_t> chunk_off; // [QD+1], per-chunk [start,end) offsets
    std::vector<uint8_t> codes;      // [N * QD]
    uint32_t N = 0, D = 0, QD = 0;
    std::vector<std::unique_ptr<float[]>> cc; // QD * (256*256) LUTs

    void load_pivots_with_offsets(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            die("open pivots");
        uint32_t nr0 = 0, nc0 = 0;
        _read_exact(f, &nr0, 4);
        _read_exact(f, &nc0, 4);
        (void)nr0;
        (void)nc0;
        uint64_t meta[4] = {0, 0, 0, 0};
        _read_exact(f, meta, sizeof(meta));

        // [0]: pivots block: expect 256 x D floats
        {
            f.seekg((std::streamoff)meta[0], std::ios::beg);
            uint32_t r = 0, c = 0;
            _read_exact(f, &r, 4);
            _read_exact(f, &c, 4);
            if (r != (uint32_t)NUM_CENTROIDS)
                die("pivots rows != 256");
            D = c;
            pivots.resize((size_t)NUM_CENTROIDS * (size_t)D);
            _read_exact(f, pivots.data(), pivots.size() * sizeof(float));
        }
        // [2]: chunkOffsets block: length = QD+1
        {
            f.seekg((std::streamoff)meta[2], std::ios::beg);
            uint32_t r = 0, c = 0;
            _read_exact(f, &r, 4);
            _read_exact(f, &c, 4);
            if (c != 1)
                die("chunkOffsets dims second != 1");
            chunk_off.resize(r);
            _read_exact(f, chunk_off.data(), (size_t)r * sizeof(uint32_t));
            if (r < 2)
                die("chunkOffsets too short");
            QD = r - 1;
        }
    }

    void load_codes_bin(const std::string &path)
    {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            die("open codes");
        uint32_t npts = 0, nch = 0;
        _read_exact(f, &npts, 4);
        _read_exact(f, &nch, 4);
        N = npts;
        if (QD && nch != QD)
        {
            std::cerr << "warning: codes nch(" << nch << ") != QD(" << QD << ") -> using nch\n";
        }
        QD = nch;
        codes.resize((size_t)N * (size_t)QD);
        _read_exact(f, codes.data(), codes.size() * sizeof(uint8_t));
    }

    void build_cc_luts()
    {
        if (chunk_off.size() != (size_t)QD + 1)
            die("chunk_off/QD mismatch");
        cc.resize(QD);
#pragma omp parallel for schedule(static)
        for (int m = 0; m < (int)QD; ++m)
        {
            auto lut = std::make_unique<float[]>((size_t)NUM_CENTROIDS * (size_t)NUM_CENTROIDS);
            const uint32_t off = chunk_off[m];
            const uint32_t len = chunk_off[m + 1] - off;
            for (int a = 0; a < NUM_CENTROIDS; ++a)
            {
                const float *A = pivots.data() + (size_t)a * D + off;
                for (int b = 0; b < NUM_CENTROIDS; ++b)
                {
                    const float *B = pivots.data() + (size_t)b * D + off;
                    double acc = 0.0;
                    for (uint32_t k = 0; k < len; ++k)
                    {
                        double t = (double)A[k] - (double)B[k];
                        acc += t * t;
                    }
                    lut[(size_t)a * NUM_CENTROIDS + b] = (float)acc;
                }
            }
            cc[m] = std::move(lut);
        }
    }

    inline float dist(uint32_t u, uint32_t v) const
    {
        const uint8_t *cu = codes.data() + (size_t)u * QD;
        const uint8_t *cv = codes.data() + (size_t)v * QD;
        double acc = 0.0;
        for (uint32_t m = 0; m < QD; ++m)
        {
            acc += (double)cc[m][(size_t)cu[m] * NUM_CENTROIDS + cv[m]];
        }
        return (float)acc;
    }

    size_t bytes_in_ram() const
    {
        size_t piv = pivots.size() * sizeof(float);
        size_t cof = chunk_off.size() * sizeof(uint32_t);
        size_t cod = codes.size() * sizeof(uint8_t);
        size_t luts = (size_t)QD * (size_t)NUM_CENTROIDS * (size_t)NUM_CENTROIDS * sizeof(float);
        return piv + cof + cod + luts;
    }
};

// ---------- Output Graph Header ----------
struct GraphHdr
{
    uint64_t isz;    // final file size
    uint32_t maxObs; // max observed degree
    uint32_t ep;     // entry point (unused)
    uint64_t nf;     // frozen points (unused)
};

// ---------- Main ----------
int main(int argc, char **argv)
{
    // CLI Check
    if (argc < 2)
    {
        std::cerr << "Usage: ./apply_pq_prune --data=base.fbin --cand-prefix=PREFIX --out=graph.mem.index "
                     "[--pivots=...] [--codes=...] [--QD=32] [--R=32] [--T=1] [--P=20] [--threads=32] "
                     "[--pval=0.05] [--reuse=1] [--safety-frac=0.05] [--memory-budget-bytes=N]\n";
        return 1;
    }
    Args a;
    parse_flags(argc, argv, a);

    // Env override check
    if (a.chunk_bytes > 0)
    {
        std::string s = std::to_string(a.chunk_bytes);
        setenv("SIGMA_CHUNK_BYTES", s.c_str(), 1);
        std::cout << "[our_pruning] Applied SIGMA_CHUNK_BYTES=" << s << " from --chunk-bytes\n";
    }
    std::cout << "[our_pruning] HIDANN_CONSTRUCTION_FADVISE=" << (construction_fadvise_enabled() ? 1 : 0)
              << " HIDANN_CONSTRUCTION_SYNC_BEFORE_DROP=" << (construction_sync_before_drop_enabled() ? 1 : 0)
              << "\n";

    if (a.threads)
        omp_set_num_threads((int)a.threads);

    // When --threads is omitted, OpenMP may still spawn more than one worker.
    // Size per-thread accumulators from the effective team size, not from args.
    const int thread_slots = omp_get_max_threads();

    if (a.data_path.empty() || a.cand_prefix.empty() || a.out_graph.empty())
        die("missing required flags");
    if (a.P == 0)
        die("please pass --P=<partitions> for mem budget");
    if (a.safety_frac < 0.0f)
        a.safety_frac = 0.0f;

    // Timers
    double t_pq_gen_s = 0.0, t_pq_load_s = 0.0, t_pq_lut_s = 0.0;

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

    // Default PQ paths
    if (a.pivots_path.empty())
        a.pivots_path = a.cand_prefix + "pq_pivots.bin";
    if (a.codes_path.empty())
        a.codes_path = a.cand_prefix + "pq_compressed.bin";

    // ---------- Generate PQ Codes (if needed) ----------
    bool need_generate = !(a.reuse && path_exists(a.pivots_path) && path_exists(a.codes_path));
    if (need_generate)
    {
        std::cout << "[our_pruning] Generating PQ codes...\n";
        float pv = a.p_val;
        if (pv <= 0.f)
            pv = std::min(1.0f, 256000.0f / (float)N_data);
        bool use_opq = false;
        std::string codebook_prefix = "";
        int num_chunks = (a.QD ? (int)a.QD : 0);
        auto tg0 = Clock::now();

        // This function handles memory internally, likely safe or handled separately.
        diskann::generate_quantized_data<float>(a.data_path.c_str(), a.pivots_path.c_str(), a.codes_path.c_str(),
                                                diskann::Metric::L2, pv, num_chunks, use_opq, codebook_prefix);
        auto tg1 = Clock::now();
        t_pq_gen_s = sec(tg0, tg1);
    }

    // ---------- Load PQ and Build LUTs ----------
    std::cout << "[our_pruning] Loading PQ codes and Pivots...\n";
    PQSymL2 pq;
    auto tl0 = Clock::now();
    pq.load_pivots_with_offsets(a.pivots_path);
    pq.load_codes_bin(a.codes_path);
    drop_file_cache(a.pivots_path);
    drop_file_cache(a.codes_path);
    auto tl1 = Clock::now();
    t_pq_load_s = sec(tl0, tl1);

    if (pq.D != (uint32_t)D_data)
        std::cerr << "warning: pivots D(" << pq.D << ") vs base D(" << D_data << ") mismatch; using pivots D.\n";

    auto tb0 = Clock::now();
    pq.build_cc_luts();
    auto tb1 = Clock::now();
    t_pq_lut_s = sec(tb0, tb1);

    const uint64_t N = pq.N;
    const uint32_t D = pq.D;
    const uint32_t QD = pq.QD;

    // ---------- Calculate Memory Budget ----------
    const long double per_node_bytes = (long double)D * 4.0L + (long double)a.R * 4.0L;
    const long double total_budget_ld = (long double)N * per_node_bytes / (long double)a.P * 2.0L;
    const uint64_t formula_budget_bytes = (uint64_t)total_budget_ld;
    const uint64_t mem_budget_bytes =
        a.memory_budget_bytes > 0 ? a.memory_budget_bytes : formula_budget_bytes;

    const size_t pq_bytes = pq.bytes_in_ram();
    const uint64_t safety = (uint64_t)(mem_budget_bytes * (long double)a.safety_frac);

    uint64_t mem_avail_for_batch = 0;
    if (mem_budget_bytes > pq_bytes + safety)
        mem_avail_for_batch = mem_budget_bytes - pq_bytes - safety;
    else
        die("PQ resident bytes exceed mem budget. Streaming optimization cannot fix this if PQ itself is too big.");

    std::cout << "[our_pruning] Memory Analysis:\n"
              << "  Budget Source: " << (a.memory_budget_bytes > 0 ? "explicit" : "P-derived")
              << " (P-derived would be " << (formula_budget_bytes / 1024.0 / 1024.0) << " MB)\n"
              << "  Total Budget: " << (mem_budget_bytes / 1024.0 / 1024.0) << " MB\n"
              << "  PQ Static:    " << (pq_bytes / 1024.0 / 1024.0) << " MB\n"
              << "  Safety:       " << (safety / 1024.0 / 1024.0) << " MB\n"
              << "  Avail RAM:    " << (mem_avail_for_batch / 1024.0 / 1024.0) << " MB\n";

    // ---------- Open Candidates (Streamed) ----------
    const std::string idx_path = a.cand_prefix + "cand.idx";
    int idx_fd = ::open(idx_path.c_str(), O_RDONLY);
    if (idx_fd < 0)
        die("open cand.idx");

    // Read Header
    CandIdxHdr ih_obj;
    if (::read(idx_fd, &ih_obj, sizeof(CandIdxHdr)) != sizeof(CandIdxHdr))
        die("read cand.idx header");

    // Check offsets
    if (ih_obj.N != N)
        std::cerr << "warning: cand.idx N(" << ih_obj.N << ") != N(" << N << ")\n";

    // Data starts after header
    off_t idx_rows_start_offset = sizeof(CandIdxHdr);

    // Open cand.ids
    const std::string ids_path = a.cand_prefix + "cand.ids";
    int ids_fd = ::open(ids_path.c_str(), O_RDONLY | O_LARGEFILE);
    if (ids_fd < 0)
        die("open cand.ids");

    // Open Output
    int out_fd = ::open(a.out_graph.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_LARGEFILE, 0644);
    if (out_fd < 0)
        die("open out");
    struct GraphHdr gh { 0ull, a.R, 0u, 0ull };
    if (::pwrite(out_fd, &gh, sizeof(gh), 0) != (ssize_t)sizeof(gh))
        die("write header");
    off_t out_off = (off_t)sizeof(gh);


    // ---------- Streaming Logic ----------
    // Determine metadata batch size
    // Allocate 10% of available memory for metadata buffer, ensuring at least 10k rows
    uint64_t meta_ram_target = (uint64_t)(mem_avail_for_batch * 0.1);
    uint64_t meta_batch_rows = meta_ram_target / sizeof(CandIdxRow);
    if (meta_batch_rows < 10000) meta_batch_rows = 10000;
    if (meta_batch_rows > 10000000) meta_batch_rows = 10000000; // Cap at 10M rows to avoid excessive latency

    // Reduce avail mem by this buffer size
    mem_avail_for_batch -= (meta_batch_rows * sizeof(CandIdxRow));
    uint64_t effective_batch_budget = (uint64_t)((long double)mem_avail_for_batch * PRUNE_BATCH_PAYLOAD_FILL);
    if (a.chunk_bytes > 0)
        effective_batch_budget = std::min(effective_batch_budget, a.chunk_bytes);
    effective_batch_budget = std::min(effective_batch_budget, PRUNE_BATCH_MAX_BYTES);
    if (effective_batch_budget == 0)
        die("effective pruning batch budget is zero");

    std::cout << "  Meta Batch:   " << meta_batch_rows << " rows ("
              << (meta_batch_rows * sizeof(CandIdxRow) / 1024.0 ) << " KB)\n";
    std::cout << "  Batch Budget: " << (effective_batch_budget / 1024.0 / 1024.0)
              << " MB (fill=" << (double)PRUNE_BATCH_PAYLOAD_FILL
              << ", cap=" << (PRUNE_BATCH_MAX_BYTES / 1024.0 / 1024.0) << " MB)\n"
              << std::flush;

    std::vector<CandIdxRow> meta_buf(meta_batch_rows);
    std::vector<uint8_t> ids_chunk;
    std::vector<uint32_t> deg;
    std::vector<uint32_t> adj;
    std::vector<uint8_t> out_chunk;

    // Helper: Pick optimal batch k within local metadata buffer
    auto pick_batch_k_local = [&](uint64_t u_start_in_buf, uint64_t n_in_buf, uint64_t u_total_base) -> uint64_t {
        uint64_t lo = 1, hi = n_in_buf, best = 1;

        // Pre-calc offsets relative to ids file
        // We need min/max offset in ids file for the candidate range
        // In local buffer meta_buf[i], .off_bytes is relative to ids stream start.

        while (lo <= hi) {
            uint64_t k = (lo + hi) >> 1;

            // Calc span
            uint64_t mn = std::numeric_limits<uint64_t>::max();
            uint64_t mx = 0;

            for(uint64_t i=0; i<k; ++i) {
                uint64_t off = meta_buf[u_start_in_buf + i].off_bytes;
                uint32_t cnt = meta_buf[u_start_in_buf + i].cnt;
                uint64_t end = off + (uint64_t)cnt * 4ull;

                if (off < mn) mn = off;
                if (end > mx) mx = end;
            }

            uint64_t span_bytes = (mx > mn) ? (mx - mn) : 0;
            uint64_t per_node = (uint64_t)a.R * 4ull + 4ull;
            uint64_t need = span_bytes + 2ull * k * per_node;

            if (need <= effective_batch_budget) {
                best = k;
                lo = k + 1;
            } else {
                hi = k - 1;
            }
        }
        return best;
    };

    // Metrics
    std::atomic<uint64_t> sum_raw_cand{0};
    std::atomic<uint64_t> sum_prune_ns{0};
    uint64_t sum_ids_io_ns = 0;
    uint64_t sum_write_io_ns = 0;
    uint32_t max_observed_deg = 0;

    auto t_all_0 = Clock::now();
    uint64_t u_global = 0;

    // Outer Loop: Metadata chunks
    while (u_global < N)
    {
        uint64_t rows_to_read = std::min((uint64_t)N - u_global, (uint64_t)meta_batch_rows);

        // Read Metadata Batch
        ssize_t bytes_to_read = rows_to_read * sizeof(CandIdxRow);
        ssize_t rn = ::pread(idx_fd, meta_buf.data(), bytes_to_read,
                             idx_rows_start_offset + (off_t)(u_global * sizeof(CandIdxRow)));
        if (rn != bytes_to_read)
             die("read meta batch failed");

        // Inner Loop: Processing within metadata batch
        uint64_t u_local = 0;

        while (u_local < rows_to_read)
        {
            uint64_t remain_in_buf = rows_to_read - u_local;

            // This relies on meta_buf[u_local...]
            uint64_t batch_k = pick_batch_k_local(u_local, remain_in_buf, u_global + u_local);

            // Identify Span
            uint64_t off_min = std::numeric_limits<uint64_t>::max();
            uint64_t off_max = 0;
            for(uint64_t i=0; i<batch_k; ++i) {
                uint64_t off = meta_buf[u_local + i].off_bytes;
                uint64_t end = off + (uint64_t)meta_buf[u_local + i].cnt * 4ull;
                if(off < off_min) off_min = off;
                if(end > off_max) off_max = end;
            }
            uint64_t span_bytes = (off_max > off_min) ? (off_max - off_min) : 0;
            if (span_bytes > SIZE_MAX) die("span too large");

            ids_chunk.resize((size_t)span_bytes);

            auto tio0 = Clock::now();
            if (span_bytes > 0) {
                size_t got = 0;
                // pread from correct offset in ids file given by off_min
                // Note: cand.ids layout is Header + IDs
                // off_bytes in struct is offset from IDs start (after header)
                while(got < (size_t)span_bytes) {
                    ssize_t rn = ::pread(ids_fd, ids_chunk.data() + got, (size_t)span_bytes - got,
                                         (off_t)sizeof(CandIdsHdr) + (off_t)off_min + (off_t)got);
                    if (rn <= 0) die("pread ids batch");
                    got += (size_t)rn;
                }
            }
            auto tio1 = Clock::now();
            sum_ids_io_ns += (uint64_t)(sec(tio0, tio1) * 1e9);

            // Output buffers
            deg.assign((size_t)batch_k, 0);
            adj.assign((size_t)batch_k * (size_t)a.R, 0);

            // Parallel Pruning
            std::vector<uint64_t> thread_prune_ns((size_t)thread_slots, 0);
            std::vector<uint64_t> thread_raw_cnt((size_t)thread_slots, 0);

            #pragma omp parallel
            {
                const int tid = omp_get_thread_num();
                std::vector<std::pair<uint32_t, float>> pool;

                #pragma omp for schedule(static)
                for (ptrdiff_t i = 0; i < (ptrdiff_t)batch_k; ++i)
                {
                    uint64_t u_curr_local = u_local + i;
                    uint64_t u_curr_global = u_global + u_curr_local;

                    const uint32_t c = meta_buf[u_curr_local].cnt;
                    const uint64_t my_off = meta_buf[u_curr_local].off_bytes;

                    // Pointer into ids_chunk
                    const size_t rel = (size_t)(my_off - off_min);
                    const uint32_t *cand = reinterpret_cast<const uint32_t *>(ids_chunk.data() + rel);

                    pool.clear();
                    pool.reserve(c);
                    for (uint32_t k = 0; k < c; ++k)
                    {
                        const uint32_t v = cand[k];
                        if (v >= N || v == (uint32_t)u_curr_global) // self-loop check
                            continue;
                        const float dv = pq.dist((uint32_t)u_curr_global, v);
                        pool.emplace_back(v, dv);
                    }

                    // Sort by distance
                    std::sort(pool.begin(), pool.end(), [](const auto &a, const auto &b) {
                        return a.second < b.second || (a.second == b.second && a.first < b.first);
                    });

                    auto tp0 = Clock::now();
                    uint32_t *out = adj.data() + (size_t)i * a.R;
                    uint32_t out_deg = 0;

                    // Heuristic Pruning (T-occlusion)
                   for (size_t pi = 0; pi < pool.size() && out_deg < a.R; ++pi)
                    {
                        const uint32_t v = pool[pi].first;
                        const float dv = pool[pi].second;
                        int oc_by = 0;
                        for (uint32_t k = 0; k < out_deg; ++k)
                        {
                            const uint32_t s = out[k];
                            const float dsv = pq.dist(s, v);
                            if (dsv <= dv && ++oc_by >= (int)a.T)
                                break;
                        }
                        if (oc_by < (int)a.T)
                            out[out_deg++] = v;
                    }
                    if (out_deg < a.R)
                    {
                         for (size_t pi = 0; pi < pool.size() && out_deg < a.R; ++pi)
                        {
                            const uint32_t v = pool[pi].first;
                            bool dup = false;
                            for (uint32_t k = 0; k < out_deg; ++k)
                                if (out[k] == v) { dup=true; break; }
                            if (!dup) out[out_deg++] = v;
                        }
                    }
                    deg[i] = out_deg;

                    auto tp1 = Clock::now();
                    thread_prune_ns[tid] += (uint64_t)(sec(tp0, tp1) * 1e9);
                    thread_raw_cnt[tid] += c;
                }
            } // omp

            // Accumulate Stats
            uint64_t batch_prune_ns = 0, batch_raw_cnt = 0;
            for (auto x : thread_prune_ns) batch_prune_ns += x;
            for (auto x : thread_raw_cnt) batch_raw_cnt += x;
            sum_prune_ns += batch_prune_ns;
            sum_raw_cand += batch_raw_cnt;

            // Serialize
            size_t batch_bytes = 0;
            for (uint64_t i = 0; i < batch_k; ++i)
                batch_bytes += sizeof(uint32_t) + (size_t)deg[i] * 4u;

            out_chunk.resize(batch_bytes);
            uint8_t *wp = out_chunk.data();
            for (uint64_t i = 0; i < batch_k; ++i)
            {
                const uint32_t d = deg[i];
                std::memcpy(wp, &d, sizeof(uint32_t));
                wp += sizeof(uint32_t);
                if (d) {
                    std::memcpy(wp, adj.data() + (size_t)i * a.R, (size_t)d * 4u);
                    wp += (size_t)d * 4u;
                }
                if (d > max_observed_deg) max_observed_deg = d;
            }

            auto tw0 = Clock::now();
            {
                size_t done = 0;
                uint64_t drop_start = (uint64_t)out_off;
                while(done < batch_bytes) {
                    size_t want = std::min(WRITE_DROP_INTERVAL, batch_bytes - done);
                    ssize_t wn = ::pwrite(out_fd, out_chunk.data() + done, want, out_off);
                    if (wn < 0) die("pwrite batch");
                    done += (size_t)wn;
                    out_off += wn;
                    uint64_t written_end = (uint64_t)out_off;
                    if (written_end - drop_start >= WRITE_DROP_INTERVAL)
                    {
                        flush_and_drop_range(out_fd, drop_start, written_end - drop_start, a.out_graph.c_str());
                        drop_start = written_end;
                    }
                }
                flush_and_drop_range(out_fd, drop_start, (uint64_t)out_off - drop_start, a.out_graph.c_str());
            }
             auto tw1 = Clock::now();
            sum_write_io_ns += (uint64_t)(sec(tw0, tw1) * 1e9);
            if (span_bytes > 0)
                fadvise_dontneed(ids_fd, (uint64_t)sizeof(CandIdsHdr) + off_min, span_bytes, ids_path.c_str());

            u_local += batch_k;
        } // End Inner Loop (local batch)

        u_global += rows_to_read;
    } // End Outer Loop (Meta Batch)

     // ---------- Finalize and Report ----------
    struct GraphHdr gh2;
    gh2.isz = (uint64_t)out_off;
    gh2.maxObs = max_observed_deg;
    gh2.ep = 0u;
    gh2.nf = 0ull;
    if (::pwrite(out_fd, &gh2, sizeof(gh2), 0) != (ssize_t)sizeof(gh2))
        die("rewrite header");
    ::fsync(out_fd);
    ::close(out_fd);
    ::close(ids_fd);
    ::close(idx_fd);

    auto t_all_1 = Clock::now();

    // Report
    double total_s = sec(t_all_0, t_all_1);
    double avg_raw = (double)sum_raw_cand / (double)N;
    double avg_prune_us = (double)sum_prune_ns / 1e3 / (double)N;
    double ids_io_s = (double)sum_ids_io_ns / 1e9;
    double write_io_s = (double)sum_write_io_ns / 1e9;

    std::cout << "PQ-pruning done (Optimized Stream).\n"
              << "  N=" << N << " D=" << D << " QD=" << QD << " R=" << a.R << " P=" << a.P << "\n"
              << "  mem_budget=" << (mem_budget_bytes / (1024.0 * 1024.0)) << " MiB"
              << " | pq_resident=" << (pq_bytes / (1024.0 * 1024.0)) << " MiB"
              << " | safety_frac=" << a.safety_frac
              << " | mem_avail_for_batch=" << (mem_avail_for_batch / (1024.0 * 1024.0)) << " MiB"
              << " | effective_batch_budget=" << (effective_batch_budget / (1024.0 * 1024.0)) << " MiB\n"
              << "  pq_gen=" << t_pq_gen_s << " s"
              << " | pq_load=" << t_pq_load_s << " s"
              << " | pq_lut=" << t_pq_lut_s << " s\n"
              << "  time_total=" << total_s << " s"
              << " | ids_io=" << ids_io_s << " s"
              << " | write_io=" << write_io_s << " s"
              << " | prune_avg=" << avg_prune_us << " us/node\n"
              << "  avg_raw_cands=" << avg_raw << "\n"
              << "  max_observed_degree=" << max_observed_deg << "\n";

    return 0;
}

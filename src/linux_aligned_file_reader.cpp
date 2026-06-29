// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "linux_aligned_file_reader.h"

#include <cassert>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include "tsl/robin_map.h"
#include "utils.h"
#ifdef SIGMA_WITH_IO_URING
#include <liburing.h>
#endif

#define MAX_EVENTS 1024

namespace
{
typedef struct io_event io_event_t;
typedef struct iocb iocb_t;

// [OPTIMIZATION] Thread-local cache for IOContext to avoid mutex contention
// We cache the context for the last used Reader instance.
// Since threads typically access one reader at a time, this yields 100% hit rate.
thread_local LinuxAlignedFileReader* t_cached_reader_ptr = nullptr;
thread_local IOContext t_cached_io_ctx;

bool sigma_reader_verbose()
{
    static const bool verbose = []() {
        const char *env = std::getenv("SIGMA_VERBOSE_IO_READER");
        return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    return verbose;
}

#ifdef SIGMA_WITH_IO_URING
void execute_io_uring(struct io_uring *ring, int fd, std::vector<AlignedRead> &read_reqs)
{
    size_t n_reqs = read_reqs.size();
    size_t processed = 0;

    while (processed < n_reqs) {
        size_t batch_size = 0;

        // Prepare batch
        while (processed + batch_size < n_reqs && batch_size < MAX_EVENTS) {
             struct io_uring_sqe *sqe = io_uring_get_sqe(ring);
             if (!sqe) break; // Ring full

             auto& req = read_reqs[processed + batch_size];

             // Clean implementation using opcode IORING_OP_READ (Requires Kernel 5.6+)
             io_uring_prep_read(sqe, fd, req.buf, req.len, req.offset);

             // We use user_data to track requests if needed, but for simple batch processing
             // strict ordering is maintained.
             io_uring_sqe_set_data(sqe, nullptr);

             batch_size++;
        }

        // Submit
        int ret = io_uring_submit(ring);
        if (ret < 0) {
            std::cerr << "io_uring_submit failed: " << -ret << std::endl;
            exit(-1);
        }

        // Wait for completions
        // We expect 'ret' completions since we submitted 'ret' requests.
        size_t submitted = ret;
        size_t completed_in_batch = 0;

        while (completed_in_batch < submitted) {
            struct io_uring_cqe *cqe;

            // Standard wait (blocks until at least one event is ready)
            ret = io_uring_wait_cqe(ring, &cqe);

            if (ret < 0) {
                std::cerr << "io_uring_wait_cqe failed: " << -ret << std::endl;
                exit(-1);
            }
            if (cqe->res < 0) {
                 std::cerr << "Async read failed: " << -cqe->res
                           << " (fd=" << fd << ")" << std::endl;
                 exit(-1);
            }

            io_uring_cqe_seen(ring, cqe);
            completed_in_batch++;
        }

        // Advance by the number of actually submitted requests
        processed += submitted;
    }
}
#endif


void execute_io(io_context_t ctx, int fd, std::vector<AlignedRead> &read_reqs, uint64_t n_retries = 0)
{
#ifdef DEBUG
    for (auto &req : read_reqs)
    {
        assert(IS_ALIGNED(req.len, 512));
        // std::cout << "request:"<<req.offset<<":"<<req.len << std::endl;
        assert(IS_ALIGNED(req.offset, 512));
        assert(IS_ALIGNED(req.buf, 512));
        // assert(malloc_usable_size(req.buf) >= req.len);
    }
#endif

    // break-up requests into chunks of size MAX_EVENTS each
    uint64_t n_iters = ROUND_UP(read_reqs.size(), MAX_EVENTS) / MAX_EVENTS;
    for (uint64_t iter = 0; iter < n_iters; iter++)
    {
        uint64_t n_ops = std::min((uint64_t)read_reqs.size() - (iter * MAX_EVENTS), (uint64_t)MAX_EVENTS);
        std::vector<iocb_t *> cbs(n_ops, nullptr);
        std::vector<io_event_t> evts(n_ops);
        std::vector<struct iocb> cb(n_ops);
        for (uint64_t j = 0; j < n_ops; j++)
        {
            io_prep_pread(cb.data() + j, fd, read_reqs[j + iter * MAX_EVENTS].buf, read_reqs[j + iter * MAX_EVENTS].len,
                          read_reqs[j + iter * MAX_EVENTS].offset);
        }

        // initialize `cbs` using `cb` array
        //

        for (uint64_t i = 0; i < n_ops; i++)
        {
            cbs[i] = cb.data() + i;
        }

        uint64_t n_tries = 0;
        while (n_tries <= n_retries)
        {
            // issue reads
            int64_t ret = io_submit(ctx, (int64_t)n_ops, cbs.data());
            // if requests didn't get accepted
            if (ret != (int64_t)n_ops)
            {
                std::cerr << "io_submit() failed; returned " << ret << ", expected=" << n_ops << ", ernno=" << errno
                          << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                std::cout << "ctx: " << ctx << "\n";
                exit(-1);
            }
            else
            {
                // wait on io_getevents
                ret = io_getevents(ctx, (int64_t)n_ops, (int64_t)n_ops, evts.data(), nullptr);
                // if requests didn't complete
                if (ret != (int64_t)n_ops)
                {
                    std::cerr << "io_getevents() failed; returned " << ret << ", expected=" << n_ops
                              << ", ernno=" << errno << "=" << ::strerror(-ret) << ", try #" << n_tries + 1;
                    exit(-1);
                }
                else
                {
                    break;
                }
            }
        }
        // disabled since req.buf could be an offset into another buf
        /*
        for (auto &req : read_reqs) {
          // corruption check
          assert(malloc_usable_size(req.buf) >= req.len);
        }
        */
    }
}
} // namespace

LinuxAlignedFileReader::LinuxAlignedFileReader()
{
    this->file_desc = -1;
    const char* env = std::getenv("SIGMA_IO_ENGINE");
    if (env && std::string(env) == "io_uring") {
#ifdef SIGMA_WITH_IO_URING
        this->engine_type = IOContext::Type::IOURING;
#else
        std::cerr << "WARNING: io_uring requested but not compiled. Falling back to libaio." << std::endl;
        this->engine_type = IOContext::Type::LIBAIO;
#endif
    } else {
        this->engine_type = IOContext::Type::LIBAIO;
    }
}

LinuxAlignedFileReader::~LinuxAlignedFileReader()
{
    int64_t ret;
    // check to make sure file_desc is closed
    ret = ::fcntl(this->file_desc, F_GETFD);
    if (ret == -1)
    {
        if (errno != EBADF)
        {
            std::cerr << "close() not called" << std::endl;
            // close file desc
            ret = ::close(this->file_desc);
            // error checks
            if (ret == -1)
            {
                std::cerr << "close() failed; returned " << ret << ", errno=" << errno << ":" << ::strerror(errno)
                          << std::endl;
            }
        }
    }
}

IOContext &LinuxAlignedFileReader::get_ctx()
{
    // [OPTIMIZATION] Fast path: Check thread-local cache
    if (t_cached_reader_ptr == this) {
        return t_cached_io_ctx;
    }

    std::unique_lock<std::mutex> lk(ctx_mut);
    // perform checks only in DEBUG mode
    if (ctx_map.find(std::this_thread::get_id()) == ctx_map.end())
    {
        throw diskann::ANNException("get_ctx() called from an unregistered thread", -1, __FUNCSIG__, __FILE__,
                                    __LINE__);
    }
    else
    {
        // Update cache
        t_cached_io_ctx = ctx_map[std::this_thread::get_id()];
        t_cached_reader_ptr = this;
        return t_cached_io_ctx;
    }
}

void LinuxAlignedFileReader::register_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    if (ctx_map.find(my_id) != ctx_map.end())
    {
        throw diskann::ANNException("register_thread() called more than once from the same thread for the same reader",
                                    -1, __FUNCSIG__, __FILE__, __LINE__);
    }
    if (this->engine_type == IOContext::Type::LIBAIO) {
        io_context_t ctx = 0;
        int ret = io_setup(MAX_EVENTS, &ctx);
        if (ret != 0)
        {
            lk.unlock();
            std::string msg;
            if (ret == -EAGAIN)
            {
                msg = "io_setup() failed with EAGAIN: Consider increasing /proc/sys/fs/aio-max-nr";
            }
            else
            {
                msg = "io_setup() failed; returned " + std::to_string(ret) + ": " + ::strerror(-ret);
            }
            throw diskann::ANNException(msg, -1, __FUNCSIG__, __FILE__, __LINE__);
        }

        IOContext io_ctx;
        io_ctx.type = IOContext::Type::LIBAIO;
        io_ctx.aio_ctx = ctx;

        if (sigma_reader_verbose())
        {
            diskann::cout << "allocating libaio ctx: " << ctx << " to thread-id:" << my_id << std::endl;
        }
        ctx_map[my_id] = io_ctx;
    } else {
#ifdef SIGMA_WITH_IO_URING
        // IOURING
        struct io_uring* ring = new struct io_uring;
        int ret = io_uring_queue_init(MAX_EVENTS, ring, 0);
        if (ret < 0) {
            lk.unlock();
            delete ring;
            throw diskann::ANNException("io_uring_queue_init failed: " + std::to_string(-ret), -1, __FUNCSIG__,
                                        __FILE__, __LINE__);
        }

        IOContext io_ctx;
        io_ctx.type = IOContext::Type::IOURING;
        io_ctx.ring = ring;

        if (sigma_reader_verbose())
        {
            diskann::cout << "allocating io_uring ring to thread-id:" << my_id << std::endl;
        }
        ctx_map[my_id] = io_ctx;
#else
        lk.unlock();
        throw diskann::ANNException("io_uring selected but not compiled", -1, __FUNCSIG__, __FILE__, __LINE__);
#endif
    }
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_thread()
{
    auto my_id = std::this_thread::get_id();
    std::unique_lock<std::mutex> lk(ctx_mut);
    assert(ctx_map.find(my_id) != ctx_map.end());

    lk.unlock();
    IOContext ctx = this->get_ctx();
    if (ctx.type == IOContext::Type::LIBAIO) {
        io_destroy(ctx.aio_ctx);
    } else {
#ifdef SIGMA_WITH_IO_URING
        io_uring_queue_exit(ctx.ring);
        delete ctx.ring;
#endif
    }
    //  assert(ret == 0);
    lk.lock();
    ctx_map.erase(my_id);

    // [OPTIMIZATION] Invalidate cache if it matches this reader
    if (t_cached_reader_ptr == this) {
        t_cached_reader_ptr = nullptr;
    }

    if (sigma_reader_verbose())
    {
        std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    }
    lk.unlock();
}

void LinuxAlignedFileReader::deregister_all_threads()
{
    std::unique_lock<std::mutex> lk(ctx_mut);
    for (auto x = ctx_map.begin(); x != ctx_map.end(); x++)
    {
        IOContext ctx = x.value();
        if (ctx.type == IOContext::Type::LIBAIO) {
            io_destroy(ctx.aio_ctx);
        } else {
#ifdef SIGMA_WITH_IO_URING
            io_uring_queue_exit(ctx.ring);
            delete ctx.ring;
#endif
        }
        //  assert(ret == 0);
        //  lk.lock();
        //  ctx_map.erase(my_id);
        //  std::cerr << "returned ctx from thread-id:" << my_id << std::endl;
    }
    ctx_map.clear();
    //  lk.unlock();
}

void LinuxAlignedFileReader::open(const std::string &fname)
{
    int flags = O_DIRECT | O_RDONLY | O_LARGEFILE;
    this->file_desc = ::open(fname.c_str(), flags);
    // error checks
    assert(this->file_desc != -1);
    if (sigma_reader_verbose())
    {
        std::cerr << "Opened file : " << fname << std::endl;
    }
}

void LinuxAlignedFileReader::close()
{
    //  int64_t ret;

    // check to make sure file_desc is closed
    ::fcntl(this->file_desc, F_GETFD);
    //  assert(ret != -1);

    ::close(this->file_desc);
    //  assert(ret != -1);
}

void LinuxAlignedFileReader::read(std::vector<AlignedRead> &read_reqs, IOContext &ctx, bool async)
{
    if (async == true)
    {
        diskann::cout << "Async currently not supported in linux." << std::endl;
    }
    assert(this->file_desc != -1);
    if (ctx.type == IOContext::Type::LIBAIO) {
        execute_io(ctx.aio_ctx, this->file_desc, read_reqs);
    } else {
#ifdef SIGMA_WITH_IO_URING
        execute_io_uring(ctx.ring, this->file_desc, read_reqs);
#endif
    }
}

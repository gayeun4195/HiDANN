// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#if defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

#include "logger.h"
#include "ann_exception.h"

// sequential cached reads
class cached_ifstream
{
  public:
    cached_ifstream()
    {
    }
    cached_ifstream(const std::string &filename, uint64_t cacheSize) : cache_size(cacheSize), cur_off(0)
    {
        reader.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        this->open(filename, cache_size);
    }
    ~cached_ifstream()
    {
        delete[] cache_buf;
        reader.close();
#if defined(__linux__)
        if (cache_drop_fd >= 0)
            ::close(cache_drop_fd);
#endif
    }

    void open(const std::string &filename, uint64_t cacheSize)
    {
        this->cur_off = 0;
        this->logical_pos = 0;
        this->last_cache_drop_offset = 0;
        this->filename_for_cache_drop = filename;

        try
        {
            reader.open(filename, std::ios::binary | std::ios::ate);
            fsize = reader.tellg();
            reader.seekg(0, std::ios::beg);
            assert(reader.is_open());
            assert(cacheSize > 0);
            cacheSize = (std::min)(cacheSize, fsize);
            this->cache_size = cacheSize;
            cache_buf = new char[cacheSize];
            reader.read(cache_buf, cacheSize);
            diskann::cout << "Opened: " << filename.c_str() << ", size: " << fsize << ", cache_size: " << cacheSize
                          << std::endl;
        }
        catch (std::system_error &e)
        {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    void enable_periodic_cache_drop(uint64_t interval_bytes)
    {
#if defined(__linux__)
        cache_drop_interval = interval_bytes;
        last_cache_drop_offset = 0;
        if (cache_drop_fd >= 0)
            ::close(cache_drop_fd);
        cache_drop_fd = ::open(filename_for_cache_drop.c_str(), O_RDONLY);
        if (cache_drop_fd < 0)
        {
            diskann::cerr << "WARNING: failed to open " << filename_for_cache_drop
                          << " for read cache drop: " << std::strerror(errno) << std::endl;
            cache_drop_interval = 0;
        }
#else
        (void)interval_bytes;
#endif
    }

    size_t get_file_size()
    {
        return fsize;
    }

    void read(char *read_buf, uint64_t n_bytes)
    {
        assert(cache_buf != nullptr);
        assert(read_buf != nullptr);

        if (n_bytes <= (cache_size - cur_off))
        {
            // case 1: cache contains all data
            memcpy(read_buf, cache_buf + cur_off, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // case 2: cache contains some data
            uint64_t cached_bytes = cache_size - cur_off;
            if (n_bytes - cached_bytes > fsize - reader.tellg())
            {
                std::stringstream stream;
                stream << "Reading beyond end of file" << std::endl;
                stream << "n_bytes: " << n_bytes << " cached_bytes: " << cached_bytes << " fsize: " << fsize
                       << " current pos:" << reader.tellg() << std::endl;
                diskann::cout << stream.str() << std::endl;
                throw diskann::ANNException(stream.str(), -1, __FUNCSIG__, __FILE__, __LINE__);
            }
            memcpy(read_buf, cache_buf + cur_off, cached_bytes);

            // go to disk and fetch more data
            reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
            // reset cur off
            cur_off = cache_size;

            uint64_t size_left = fsize - reader.tellg();

            if (size_left >= cache_size)
            {
                reader.read(cache_buf, cache_size);
                cur_off = 0;
            }
            // note that if size_left < cache_size, then cur_off = cache_size,
            // so subsequent reads will all be directly from file
        }
        logical_pos += n_bytes;
        maybe_drop_read_cache();
    }

  private:
    void maybe_drop_read_cache()
    {
#if defined(__linux__)
        if (cache_drop_fd < 0 || cache_drop_interval == 0 ||
            logical_pos - last_cache_drop_offset < cache_drop_interval)
        {
            return;
        }

        uint64_t drop_len = logical_pos - last_cache_drop_offset;
        int ret = ::posix_fadvise(cache_drop_fd, (off_t)last_cache_drop_offset, (off_t)drop_len,
                                  POSIX_FADV_DONTNEED);
        if (ret != 0 && !cache_drop_warned)
        {
            diskann::cerr << "WARNING: posix_fadvise(DONTNEED) failed for " << filename_for_cache_drop
                          << ": " << std::strerror(ret) << std::endl;
            cache_drop_warned = true;
        }
        last_cache_drop_offset = logical_pos;
#endif
    }

    // underlying ifstream
    std::ifstream reader;
    std::string filename_for_cache_drop;
    // # bytes to cache in one shot read
    uint64_t cache_size = 0;
    // underlying buf for cache
    char *cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;
    // file size
    uint64_t fsize = 0;
    uint64_t logical_pos = 0;
    uint64_t cache_drop_interval = 0;
    uint64_t last_cache_drop_offset = 0;
    bool cache_drop_warned = false;
#if defined(__linux__)
    int cache_drop_fd = -1;
#endif
};

// sequential cached writes
class cached_ofstream
{
  public:
    cached_ofstream(const std::string &filename, uint64_t cache_size)
        : filename_for_cache_drop(filename), cache_size(cache_size), cur_off(0)
    {
        writer.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        try
        {
            writer.open(filename, std::ios::binary);
            assert(writer.is_open());
            assert(cache_size > 0);
            cache_buf = new char[cache_size];
            diskann::cout << "Opened: " << filename.c_str() << ", cache_size: " << cache_size << std::endl;
        }
        catch (std::system_error &e)
        {
            throw diskann::FileException(filename, e, __FUNCSIG__, __FILE__, __LINE__);
        }
    }

    ~cached_ofstream()
    {
        this->close();
    }

    void close()
    {
        // dump any remaining data in memory
        if (cur_off > 0)
        {
            this->flush_cache();
        }

        if (cache_buf != nullptr)
        {
            delete[] cache_buf;
            cache_buf = nullptr;
        }

        if (writer.is_open())
            writer.close();
#if defined(__linux__)
        if (cache_drop_fd >= 0)
        {
            ::close(cache_drop_fd);
            cache_drop_fd = -1;
        }
#endif
        diskann::cout << "Finished writing " << fsize << "B" << std::endl;
    }

    void enable_periodic_cache_drop(uint64_t interval_bytes, bool sync_before_drop = true)
    {
#if defined(__linux__)
        cache_drop_interval = interval_bytes;
        cache_drop_sync_before_drop = sync_before_drop;
        last_cache_drop_offset = 0;
        if (cache_drop_fd >= 0)
            ::close(cache_drop_fd);
        cache_drop_fd = ::open(filename_for_cache_drop.c_str(), O_WRONLY);
        if (cache_drop_fd < 0)
        {
            diskann::cerr << "WARNING: failed to open " << filename_for_cache_drop
                          << " for write cache drop: " << std::strerror(errno) << std::endl;
            cache_drop_interval = 0;
        }
#else
        (void)interval_bytes;
        (void)sync_before_drop;
#endif
    }

    size_t get_file_size()
    {
        return fsize;
    }
    // writes n_bytes from write_buf to the underlying ofstream/cache
    void write(char *write_buf, uint64_t n_bytes)
    {
        assert(cache_buf != nullptr);
        if (n_bytes <= (cache_size - cur_off))
        {
            // case 1: cache can take all data
            memcpy(cache_buf + cur_off, write_buf, n_bytes);
            cur_off += n_bytes;
        }
        else
        {
            // case 2: cache cant take all data
            // go to disk and write existing cache data
            writer.write(cache_buf, cur_off);
            fsize += cur_off;
            // write the new data to disk
            writer.write(write_buf, n_bytes);
            fsize += n_bytes;
            // memset all cache data and reset cur_off
            memset(cache_buf, 0, cache_size);
            cur_off = 0;
            maybe_drop_written_cache();
        }
    }

    void flush_cache()
    {
        assert(cache_buf != nullptr);
        writer.write(cache_buf, cur_off);
        fsize += cur_off;
        memset(cache_buf, 0, cache_size);
        cur_off = 0;
        maybe_drop_written_cache();
    }

    void reset()
    {
        flush_cache();
        writer.seekp(0);
    }

  private:
    void maybe_drop_written_cache()
    {
#if defined(__linux__)
        if (cache_drop_fd < 0 || cache_drop_interval == 0 ||
            fsize - last_cache_drop_offset < cache_drop_interval)
        {
            return;
        }

        writer.flush();
        if (cache_drop_sync_before_drop && ::fdatasync(cache_drop_fd) != 0 && !cache_drop_warned)
        {
            diskann::cerr << "WARNING: fdatasync failed for " << filename_for_cache_drop
                          << ": " << std::strerror(errno) << std::endl;
            cache_drop_warned = true;
        }

        uint64_t drop_len = fsize - last_cache_drop_offset;
        int ret = ::posix_fadvise(cache_drop_fd, (off_t)last_cache_drop_offset, (off_t)drop_len,
                                  POSIX_FADV_DONTNEED);
        if (ret != 0 && !cache_drop_warned)
        {
            diskann::cerr << "WARNING: posix_fadvise(DONTNEED) failed for " << filename_for_cache_drop
                          << ": " << std::strerror(ret) << std::endl;
            cache_drop_warned = true;
        }
        last_cache_drop_offset = fsize;
#endif
    }

    // underlying ofstream
    std::ofstream writer;
    std::string filename_for_cache_drop;
    // # bytes to cache for one shot write
    uint64_t cache_size = 0;
    // underlying buf for cache
    char *cache_buf = nullptr;
    // offset into cache_buf for cur_pos
    uint64_t cur_off = 0;

    // file size
    uint64_t fsize = 0;
    uint64_t cache_drop_interval = 0;
    uint64_t last_cache_drop_offset = 0;
    bool cache_drop_sync_before_drop = true;
    bool cache_drop_warned = false;
#if defined(__linux__)
    int cache_drop_fd = -1;
#endif
};

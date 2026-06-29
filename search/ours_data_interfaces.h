#pragma once
#include <vector>
#include <cstdint>
#include <memory>
#include <string>
#include "aligned_file_reader.h"


namespace diskann {

struct DiskIndexMetadata {
    uint64_t num_points = 0;
    uint64_t data_dim = 0;
    uint64_t medoid = 0;
    uint64_t max_node_len = 0;
    uint64_t nnodes_per_sector = 0;
    uint64_t frozen_num = 0;
    uint64_t frozen_loc = 0;
    uint64_t append_reorder_data = 0;
};

// Read the metadata block written by create_disk_layout/save_bin<uint64_t>.
// This is the validated format used by the active COMBINED rerun pipeline.
bool load_disk_index_metadata(const std::string& filename, DiskIndexMetadata& metadata, std::string& error);

// Abstract base class for Graph Data Access
class GraphSource {
public:
    virtual ~GraphSource() = default;

    virtual void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t num_points) = 0;
    virtual uint32_t get_max_degree() = 0;

    // Translate Node ID to Read Request (Offset, Length)
    // Returns number of bytes expected
    virtual void prepare_node_read(uint32_t node_id, AlignedRead& req, char* buf) = 0;

    // Parse the buffer read from disk to extract neighbor list
    // Returns number of neighbors
    virtual uint32_t parse_neighbors(uint32_t node_id, char* buf, std::vector<uint32_t>& neighbors) = 0;

    // Get pointer to neighbors in buffer without copy
    // Returns pointer to neighbors, sets size to number of neighbors
    virtual const uint32_t* get_neighbors_ptr(uint32_t node_id, char* buf, uint32_t& size) = 0;

    // Returns the required buffer size for reading a single node
    virtual uint32_t get_read_buffer_size() = 0;
};

// Abstract base class for Vector Data Access
class VectorSource {
public:
    virtual ~VectorSource() = default;

    virtual void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t dim, uint64_t num_points) = 0;

    // Prepare batch read requests
    virtual void prepare_batch_read(const std::vector<uint32_t>& ids,
                                    const std::vector<char*>& bufs,
                                    std::vector<AlignedRead>& reqs) = 0;

    // Get detailed vector coordinates from buffer
    // Useful if vector is interleaved or needs decoding (though usually it's just raw copy)
    // Get detailed vector coordinates from buffer
    // Useful if vector is interleaved or needs decoding (though usually it's just raw copy)
    virtual const char* get_vector_ptr(uint32_t node_id, char* buf) = 0;

    // Returns the required buffer size for reading a single vector
    virtual uint32_t get_read_buffer_size() = 0;
};

// ---------------------------------------------------------
// Implementation 1: Standard DiskANN Combined Layout
// ---------------------------------------------------------
class DiskANNGraphSource final : public GraphSource {
    std::shared_ptr<AlignedFileReader> _reader;
    uint64_t _max_node_len;
    uint32_t _max_degree;
    uint64_t _dim;
    uint64_t _disk_bytes_per_point;
    uint32_t _nnodes_per_sector;
public:
    DiskANNGraphSource(uint64_t max_node_len, uint32_t max_degree, uint64_t dim, uint64_t disk_bytes_per_point);
    void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t num_points) override;
    uint32_t get_max_degree() override;
    void prepare_node_read(uint32_t node_id, AlignedRead& req, char* buf) override;
    uint32_t parse_neighbors(uint32_t node_id, char* buf, std::vector<uint32_t>& neighbors) override;
    const uint32_t* get_neighbors_ptr(uint32_t node_id, char* buf, uint32_t& size) override;
    uint32_t get_read_buffer_size() override;
};

class DiskANNVectorSource final : public VectorSource {
    std::shared_ptr<AlignedFileReader> _reader;
    uint64_t _max_node_len;
    uint64_t _dim;
    uint64_t _aligned_dim;
    uint32_t _nnodes_per_sector; // Added
public:
    DiskANNVectorSource(uint64_t max_node_len, uint64_t dim);
    void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t dim, uint64_t num_points) override;
    void prepare_batch_read(const std::vector<uint32_t>& ids, const std::vector<char*>& bufs, std::vector<AlignedRead>& reqs) override;
    const char* get_vector_ptr(uint32_t node_id, char* buf) override;
    uint32_t get_read_buffer_size() override;
};

// ---------------------------------------------------------
// Implementation 2: Separate Layout (Fixed-R Graph + Bin Vector)
// ---------------------------------------------------------
// NOTE:
//   This path is kept for future experiments, but the active rerun scripts use
//   the COMBINED DiskANN-compatible layout only. Treat the separate layout as
//   experimental unless it is revalidated end-to-end.
class FixedGraphSource : public GraphSource {
    std::shared_ptr<AlignedFileReader> _reader;
    uint32_t _max_degree; // R
    uint64_t _node_size_bytes; // sizeof(uint32_t) + R * sizeof(uint32_t)
public:
    FixedGraphSource(uint32_t max_degree);
    void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t num_points) override;
    uint32_t get_max_degree() override;
    void prepare_node_read(uint32_t node_id, AlignedRead& req, char* buf) override;
    uint32_t parse_neighbors(uint32_t node_id, char* buf, std::vector<uint32_t>& neighbors) override;
    const uint32_t* get_neighbors_ptr(uint32_t node_id, char* buf, uint32_t& size) override;
    uint32_t get_read_buffer_size() override;
};

class BinVectorSource : public VectorSource {
    std::shared_ptr<AlignedFileReader> _reader;
    uint64_t _dim;
    uint64_t _aligned_dim;
    uint64_t _vec_size_bytes;
    uint64_t _header_size; // usually 8 bytes (num) + 8 bytes (dim)
public:
    BinVectorSource();
    void load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t dim, uint64_t num_points) override;
    void prepare_batch_read(const std::vector<uint32_t>& ids, const std::vector<char*>& bufs, std::vector<AlignedRead>& reqs) override;
    const char* get_vector_ptr(uint32_t node_id, char* buf) override;
    uint32_t get_read_buffer_size() override;
};


} // namespace diskann

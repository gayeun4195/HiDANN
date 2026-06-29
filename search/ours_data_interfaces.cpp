#include "ours_data_interfaces.h"
#include <cmath>
#include "utils.h"
#include "defaults.h"

namespace diskann {

bool load_disk_index_metadata(const std::string& filename, DiskIndexMetadata& metadata, std::string& error) {
    error.clear();

    uint64_t* meta_ptr = nullptr;
    size_t meta_count = 0;
    size_t meta_dim = 0;

    try {
        diskann::load_bin<uint64_t>(filename, meta_ptr, meta_count, meta_dim);
    } catch (const std::exception& e) {
        error = e.what();
        return false;
    }

    std::unique_ptr<uint64_t[]> meta_guard(meta_ptr);

    if (meta_dim != 1) {
        error = "metadata dim must be 1, got " + std::to_string(meta_dim);
        return false;
    }
    if (meta_count < 4) {
        error = "metadata must contain at least 4 uint64 entries, got " + std::to_string(meta_count);
        return false;
    }

    DiskIndexMetadata parsed;
    parsed.num_points = meta_ptr[0];
    parsed.data_dim = meta_ptr[1];
    parsed.medoid = meta_ptr[2];
    parsed.max_node_len = meta_ptr[3];
    if (meta_count > 4) parsed.nnodes_per_sector = meta_ptr[4];
    if (meta_count > 5) parsed.frozen_num = meta_ptr[5];
    if (meta_count > 6) parsed.frozen_loc = meta_ptr[6];
    if (meta_count > 7) parsed.append_reorder_data = meta_ptr[7];

    if (parsed.num_points == 0) {
        error = "num_points must be > 0";
        return false;
    }
    if (parsed.data_dim == 0) {
        error = "data_dim must be > 0";
        return false;
    }
    if (parsed.max_node_len == 0) {
        error = "max_node_len must be > 0";
        return false;
    }
    if (parsed.medoid >= parsed.num_points) {
        error = "medoid " + std::to_string(parsed.medoid) + " is out of range for num_points " +
                std::to_string(parsed.num_points);
        return false;
    }

    metadata = parsed;
    return true;
}

// =========================================================
// DiskANNGraphSource
// =========================================================

DiskANNGraphSource::DiskANNGraphSource(uint64_t max_node_len, uint32_t max_degree, uint64_t dim,
                                       uint64_t disk_bytes_per_point)
    : _max_node_len(max_node_len), _max_degree(max_degree), _dim(dim) {
    _disk_bytes_per_point = disk_bytes_per_point;
    if (max_node_len <= 4096) {
        _nnodes_per_sector = 4096 / max_node_len;
    } else {
        _nnodes_per_sector = 0; // Indicator for Large Node Mode
    }
}

void DiskANNGraphSource::load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t num_points) {
    _reader = reader;
    // Reader is assumed to be open or we rely on it being managed externally.
    // In SigmaSearch, reader is passed.
}

uint32_t DiskANNGraphSource::get_max_degree() {
    return _max_degree;
}

void DiskANNGraphSource::prepare_node_read(uint32_t node_id, AlignedRead& req, char* buf) {
    uint64_t abs_offset;
    if (_nnodes_per_sector > 0) {
        // Low Dim: Packed in Sectors
        uint64_t sector_idx = (uint64_t)node_id / _nnodes_per_sector;
        uint64_t idx_in_sector = (uint64_t)node_id % _nnodes_per_sector;
        abs_offset = (sector_idx + 1) * 4096 + idx_in_sector * _max_node_len;
    } else {
        // High Dim: Aligned to Sectors (1+ sectors per node)
        // Offset = Header + ID * AlignedSize
        uint64_t aligned_node_len = (_max_node_len + 4095) / 4096 * 4096;
        abs_offset = 4096 + (uint64_t)node_id * aligned_node_len;
    }

    uint64_t sector_start = (abs_offset / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
    uint64_t sector_end = ((abs_offset + _max_node_len + defaults::SECTOR_LEN - 1) / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;

    req.offset = sector_start;
    req.len = (uint32_t)(sector_end - sector_start);
    req.buf = buf;
}

uint32_t DiskANNGraphSource::parse_neighbors(uint32_t node_id, char* buf, std::vector<uint32_t>& neighbors) {
    uint64_t abs_offset;
    if (_nnodes_per_sector > 0) {
         uint64_t sector_idx = (uint64_t)node_id / _nnodes_per_sector;
         uint64_t idx_in_sector = (uint64_t)node_id % _nnodes_per_sector;
         abs_offset = (sector_idx + 1) * 4096 + idx_in_sector * _max_node_len;
    } else {
         uint64_t aligned_node_len = (_max_node_len + 4095) / 4096 * 4096;
         abs_offset = 4096 + (uint64_t)node_id * aligned_node_len;
    }

    uint64_t sector_start = (abs_offset / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
    uint64_t buf_offset = abs_offset - sector_start;

    char* node_buf = buf + buf_offset;

    // DiskANN's on-disk node layout stores raw coordinates, not aligned coordinates.
    char* ptr = node_buf + _disk_bytes_per_point;
    uint32_t k = *((uint32_t*)ptr);
    ptr += sizeof(uint32_t);

    neighbors.clear();
    if (k > _max_degree) {
         k = _max_degree;
    }

    uint32_t* nbrs = (uint32_t*)ptr;
    for(uint32_t i=0; i<k; ++i) {
        neighbors.push_back(nbrs[i]);
    }
    return k;
}

const uint32_t* DiskANNGraphSource::get_neighbors_ptr(uint32_t node_id, char* buf, uint32_t& size) {
    uint64_t offset_in_buf;
    if (_nnodes_per_sector > 0) {
         offset_in_buf = (uint64_t)(node_id % _nnodes_per_sector) * _max_node_len;
    } else {
         offset_in_buf = 0;
    }

    char* node_buf = buf + offset_in_buf;

    // DiskANN's on-disk node layout stores raw coordinates, not aligned coordinates.
    char* ptr = node_buf + _disk_bytes_per_point;
    uint32_t k = *((uint32_t*)ptr);
    ptr += sizeof(uint32_t);

    if (k > _max_degree) {
         k = _max_degree;
    }
    size = k;
    return (const uint32_t*)ptr;
}

uint32_t DiskANNGraphSource::get_read_buffer_size() {
    return (uint32_t)(ROUND_UP(_max_node_len + defaults::SECTOR_LEN, defaults::SECTOR_LEN));
}

// =========================================================
// DiskANNVectorSource
// =========================================================

DiskANNVectorSource::DiskANNVectorSource(uint64_t max_node_len, uint64_t dim)
    : _max_node_len(max_node_len), _dim(dim) {
    _aligned_dim = ROUND_UP(dim, 8);
    if (max_node_len <= 4096) {
        _nnodes_per_sector = 4096 / max_node_len;
    } else {
        _nnodes_per_sector = 0;
    }
}

void DiskANNVectorSource::load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t dim, uint64_t num_points) {
    _reader = reader;
}

void DiskANNVectorSource::prepare_batch_read(const std::vector<uint32_t>& ids, const std::vector<char*>& bufs, std::vector<AlignedRead>& reqs) {
    reqs.clear();
    for(size_t i=0; i<ids.size(); ++i) {
        uint32_t node_id = ids[i];

        uint64_t abs_offset;
        if (_nnodes_per_sector > 0) {
            uint64_t sector_idx = (uint64_t)node_id / _nnodes_per_sector;
            uint64_t idx_in_sector = (uint64_t)node_id % _nnodes_per_sector;
            abs_offset = (sector_idx + 1) * 4096 + idx_in_sector * _max_node_len;
        } else {
            uint64_t aligned_node_len = (_max_node_len + 4095) / 4096 * 4096;
            abs_offset = 4096 + (uint64_t)node_id * aligned_node_len;
        }

        uint64_t sector_start = (abs_offset / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
        uint64_t sector_end = ((abs_offset + _max_node_len + defaults::SECTOR_LEN - 1) / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;

        AlignedRead req;
        req.offset = sector_start;
        req.len = (uint32_t)(sector_end - sector_start);
        req.buf = bufs[i];
        reqs.push_back(req);
    }
}

const char* DiskANNVectorSource::get_vector_ptr(uint32_t node_id, char* buf) {
    // DiskANN Layout: Vector at START of node
    uint64_t abs_offset;
    if (_nnodes_per_sector > 0) {
        uint64_t sector_idx = (uint64_t)node_id / _nnodes_per_sector;
        uint64_t idx_in_sector = (uint64_t)node_id % _nnodes_per_sector;
        abs_offset = (sector_idx + 1) * 4096 + idx_in_sector * _max_node_len;
    } else {
        uint64_t aligned_node_len = (_max_node_len + 4095) / 4096 * 4096;
        abs_offset = 4096 + (uint64_t)node_id * aligned_node_len;
    }

    uint64_t sector_start = (abs_offset / defaults::SECTOR_LEN) * defaults::SECTOR_LEN;
    uint64_t buf_offset = abs_offset - sector_start;

    return buf + buf_offset;
}

uint32_t DiskANNVectorSource::get_read_buffer_size() {
    // Max bytes needed: Node Length + up to 1 full sector alignment waste
    // If aligned, len = NodeLen.
    // Misaligned worst case: touches Ceil((NodeLen)/Sector) + 1 sectors?
    // Formula: RoundUp(NodeLen + SectorLen, SectorLen) should be safe upper bound.
    // Actually simpler: Max Read Len = RoundUp(NodeLen + 2*SectorLen, SectorLen) is super safe but wasteful.
    // Let's use the calc matches logic:
    // Max Len = Ceil(Offset + NodeLen) - Floor(Offset).
    // Max value is when Offset is -1 from sector boundary? No.
    // Max length is when it spans across sectors.
    // Approx: NodeLen + SectorLen (for alignment padding)
    return (uint32_t)(ROUND_UP(_max_node_len + defaults::SECTOR_LEN, defaults::SECTOR_LEN));
}

// =========================================================
// FixedGraphSource
// =========================================================
FixedGraphSource::FixedGraphSource(uint32_t max_degree)
    : _max_degree(max_degree) {
    _node_size_bytes = sizeof(uint32_t) + _max_degree * sizeof(uint32_t); // Count + Neighbors
}

void FixedGraphSource::load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t num_points) {
    _reader = reader;
    // Assuming simple file opening managed by reader
    // Or we might need to open separate file descriptor?
    // AlignedFileReader usually manages one file.
    // If Separate files are used, we need separate readers!
    // SigmaSearch will likely instantiate separate readers or manage files.
    // For now assume reader is dedicated.
}

uint32_t FixedGraphSource::get_max_degree() { return _max_degree; }

void FixedGraphSource::prepare_node_read(uint32_t node_id, AlignedRead& req, char* buf) {
    // Header is usually storing meta. Let's assume 0 offset for now or standard diskann graph header?
    // "Modified mem.index variant: [Header][Node 0]..."
    // Header usually: u64 points, u64 max_degree?
    // Let's assume 8+8 bytes header.
    uint64_t header_size = 16;

    // Just calculate offset
    uint64_t offset = header_size + (uint64_t)node_id * _node_size_bytes;

    // We need to align read to SECTOR!
    // This is tricky for GraphSource if it's not sector aligned naturally.
    // We might need to read a sector containing the data.
    // But fixed graph nodes might cross sector boundaries.
    // AlignedFileReader handles basic alignment if we give it offset/len? No, it often expects sector aligned offset.
    // Safe approach: align offset down to 512/4096, adjust buffer pointer, read enough len.

    // Let's implement sector-aligned read logic:
    uint64_t align = 512; // Use safetish alignment
    uint64_t aligned_off = (offset / align) * align;
    uint64_t diff = offset - aligned_off;
    uint64_t len = diff + _node_size_bytes;
    len = ROUND_UP(len, align);

    req.offset = aligned_off;
    req.len = (uint32_t)len;
    // req.buf must be the start of buffer. Caller must handle 'diff' offset after reading.
    req.buf = buf;
}

uint32_t FixedGraphSource::parse_neighbors(uint32_t node_id, char* buf, std::vector<uint32_t>& neighbors) {
    // Need to re-calculate offset diff because 'buf' is what we read into (sector aligned)
    // We need 'diff' which is node_id specific.
    // This implies 'buf' passed here is the same 'req.buf'.

    uint64_t header_size = 16;
    uint64_t offset = header_size + (uint64_t)node_id * _node_size_bytes;
    uint64_t align = 512;
    uint64_t aligned_off = (offset / align) * align;
    uint64_t diff = offset - aligned_off;

    char* node_ptr = buf + diff;
    uint32_t count = *((uint32_t*)node_ptr);
    uint32_t* nbrs = (uint32_t*)(node_ptr + 4);

    neighbors.clear();
    // Validate count
    if(count > _max_degree) count = _max_degree; // Safety
    for(uint32_t i=0; i<count; ++i) {
        neighbors.push_back(nbrs[i]);
    }
    return count;
}

const uint32_t* FixedGraphSource::get_neighbors_ptr(uint32_t node_id, char* buf, uint32_t& size) {
    uint64_t header_size = 16;
    uint64_t offset = header_size + (uint64_t)node_id * _node_size_bytes;
    uint64_t align = 512;
    uint64_t aligned_off = (offset / align) * align;
    uint64_t diff = offset - aligned_off;

    char* node_ptr = buf + diff;
    uint32_t count = *((uint32_t*)node_ptr);

    if(count > _max_degree) count = _max_degree;
    size = count;
    return (const uint32_t*)(node_ptr + 4);
}

uint32_t FixedGraphSource::get_read_buffer_size() {
    return (uint32_t)(ROUND_UP(_node_size_bytes + 512, 512));
}


// =========================================================
// BinVectorSource
// =========================================================
BinVectorSource::BinVectorSource() : _header_size(8) {} // num(4)+dim(4) usually for .bin? Or 4+4?
// Standard .bin from big-ann-benchmarks: int32 num, int32 dim. = 8 bytes.

void BinVectorSource::load(const std::string& filename, std::shared_ptr<AlignedFileReader> reader, uint64_t dim, uint64_t num_points) {
    _reader = reader;
    _dim = dim;
    // vec_size = dim * 4 (float)
    _vec_size_bytes = _dim * 4;
}

void BinVectorSource::prepare_batch_read(const std::vector<uint32_t>& ids, const std::vector<char*>& bufs, std::vector<AlignedRead>& reqs) {
    reqs.clear();
    // Optimize: sort logic? No, just generate requests.
    // .bin layout: [num:4][dim:4][vec0][vec1]...

    for(size_t i=0; i<ids.size(); ++i) {
        uint32_t id = ids[i];
        uint64_t offset = 8 + (uint64_t)id * _vec_size_bytes;

        // Align
        uint64_t align = 512;
        uint64_t aligned_off = (offset / align) * align;
        uint64_t diff = offset - aligned_off;
        uint64_t len = ROUND_UP(diff + _vec_size_bytes, align);

        reqs.emplace_back(aligned_off, (uint32_t)len, bufs[i]);
    }
}

const char* BinVectorSource::get_vector_ptr(uint32_t node_id, char* buf) {
    uint64_t offset = 8 + (uint64_t)node_id * _vec_size_bytes;
    uint64_t align = 512;
    uint64_t aligned_off = (offset / align) * align;
    uint64_t diff = offset - aligned_off;
    return buf + diff;
}

uint32_t BinVectorSource::get_read_buffer_size() {
    return (uint32_t)ROUND_UP(_vec_size_bytes, 512);
}

} // namespace diskann

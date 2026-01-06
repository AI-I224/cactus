#include "graph.h"
#include "../kernel/kernel.h"
#include <fstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>

namespace GraphFile {
    
    void save_node(CactusGraph& graph, size_t node_id, const std::string& filename) {
        graph.execute();
        void* data = graph.get_output(node_id);

        const auto& buffer = graph.get_output_buffer(node_id);
        const auto& shape = buffer.shape;
        Precision precision = buffer.precision;

        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }

        uint32_t ndim = static_cast<uint32_t>(shape.size());
        file.write(reinterpret_cast<const char*>(&ndim), sizeof(ndim));

        for (size_t dim : shape) {
            uint64_t dim_val = static_cast<uint64_t>(dim);
            file.write(reinterpret_cast<const char*>(&dim_val), sizeof(dim_val));
        }

        uint32_t prec_val = static_cast<uint32_t>(precision);
        file.write(reinterpret_cast<const char*>(&prec_val), sizeof(prec_val));

        size_t total_elements = 1;
        for (size_t dim : shape) {
            total_elements *= dim;
        }

        size_t element_size = PrecisionTraits::size_of(precision);
        size_t byte_size = total_elements * element_size;
        uint64_t size_val = static_cast<uint64_t>(byte_size);
        file.write(reinterpret_cast<const char*>(&size_val), sizeof(size_val));

        if (precision == Precision::INT8) {
            uint32_t group_size = buffer.is_grouped_int8() ? static_cast<uint32_t>(buffer.group_size) : 0;
            uint64_t num_groups = buffer.is_grouped_int8() ? static_cast<uint64_t>(buffer.num_groups) : 0;
            file.write(reinterpret_cast<const char*>(&group_size), sizeof(group_size));
            file.write(reinterpret_cast<const char*>(&num_groups), sizeof(num_groups));
        }

        file.write(static_cast<const char*>(data), byte_size);

        if (precision == Precision::INT8 && buffer.is_grouped_int8() && buffer.scales_data) {
            size_t N = shape.size() >= 1 ? shape[0] : 1;
            size_t scales_bytes = N * buffer.num_groups * sizeof(__fp16);
            file.write(static_cast<const char*>(buffer.scales_data), scales_bytes);
        }

        if (!file) {
            throw std::runtime_error("Error writing node data to file: " + filename);
        }
    }
    
    LoadedNode load_into_graph(CactusGraph& graph, const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }

        uint32_t ndim;
        file.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));

        std::vector<size_t> shape(ndim);
        for (uint32_t i = 0; i < ndim; i++) {
            uint64_t dim_val;
            file.read(reinterpret_cast<char*>(&dim_val), sizeof(dim_val));
            shape[i] = static_cast<size_t>(dim_val);
        }

        uint32_t prec_val;
        file.read(reinterpret_cast<char*>(&prec_val), sizeof(prec_val));
        Precision precision = static_cast<Precision>(prec_val);

        uint64_t size_val;
        file.read(reinterpret_cast<char*>(&size_val), sizeof(size_val));
        size_t byte_size = static_cast<size_t>(size_val);

        size_t group_size = 0;
        size_t num_groups = 0;
        if (precision == Precision::INT8) {
            uint32_t gs;
            uint64_t ng;
            file.read(reinterpret_cast<char*>(&gs), sizeof(gs));
            file.read(reinterpret_cast<char*>(&ng), sizeof(ng));
            group_size = static_cast<size_t>(gs);
            num_groups = static_cast<size_t>(ng);
        }

        if (!file) {
            throw std::runtime_error("Error reading file header: " + filename);
        }

        // Read weights
        std::vector<char> buffer(byte_size);
        file.read(buffer.data(), byte_size);

        if (!file || file.gcount() != static_cast<std::streamsize>(byte_size)) {
            throw std::runtime_error("Error reading node data: " + filename);
        }

        size_t node_id = graph.input(shape, precision);
        graph.set_input(node_id, buffer.data(), precision);

        if (precision == Precision::INT8 && group_size > 0 && num_groups > 0) {
            size_t N = shape.size() >= 1 ? shape[0] : 1;
            size_t scales_bytes = N * num_groups * sizeof(__fp16);
            std::vector<char> scales_buffer(scales_bytes);
            file.read(scales_buffer.data(), scales_bytes);

            auto& node_buffer = graph.nodes_[graph.node_index_map_.at(node_id)]->output_buffer;
            node_buffer.owned_scales = std::make_unique<char[]>(scales_bytes);
            std::memcpy(node_buffer.owned_scales.get(), scales_buffer.data(), scales_bytes);
            node_buffer.set_grouped_scales(group_size, num_groups, node_buffer.owned_scales.get());
        }

        return {node_id, shape, precision, byte_size};
    }

    MappedFile mmap_load(const std::string& filename);
}

GraphFile::MappedFile::MappedFile(const std::string& filename) 
    : fd_(-1), mapped_data_(nullptr), file_size_(0), data_offset_(0) {
    fd_ = open(filename.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("Cannot open file for mapping: " + filename);
    }
    
    struct stat st;
    if (fstat(fd_, &st) == -1) {
        close(fd_);
        throw std::runtime_error("Cannot get file size: " + filename);
    }
    file_size_ = st.st_size;
    
    mapped_data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("Cannot map file: " + filename);
    }

    madvise(mapped_data_, file_size_, MADV_WILLNEED);

    close(fd_);
    fd_ = -1;

    parse_header();
}

GraphFile::MappedFile::~MappedFile() {
    if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
        madvise(mapped_data_, file_size_, MADV_DONTNEED);
        munmap(mapped_data_, file_size_);
        mapped_data_ = nullptr;
    }
    if (fd_ != -1) {
        close(fd_);
        fd_ = -1;
    }
}

GraphFile::MappedFile::MappedFile(MappedFile&& other) noexcept
    : fd_(other.fd_), mapped_data_(other.mapped_data_), file_size_(other.file_size_),
      data_offset_(other.data_offset_), shape_(std::move(other.shape_)),
      precision_(other.precision_), byte_size_(other.byte_size_),
      group_size_(other.group_size_), num_groups_(other.num_groups_),
      scales_offset_(other.scales_offset_), is_int4_(other.is_int4_),
      unpacked_int4_data_(std::move(other.unpacked_int4_data_)) {
    other.fd_ = -1;
    other.mapped_data_ = nullptr;
    other.file_size_ = 0;
    other.is_int4_ = false;
}

GraphFile::MappedFile& GraphFile::MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        if (mapped_data_ != nullptr && mapped_data_ != MAP_FAILED) {
            munmap(mapped_data_, file_size_);
        }
        if (fd_ != -1) {
            close(fd_);
        }

        fd_ = other.fd_;
        mapped_data_ = other.mapped_data_;
        file_size_ = other.file_size_;
        data_offset_ = other.data_offset_;
        shape_ = std::move(other.shape_);
        precision_ = other.precision_;
        byte_size_ = other.byte_size_;
        group_size_ = other.group_size_;
        num_groups_ = other.num_groups_;
        scales_offset_ = other.scales_offset_;
        is_int4_ = other.is_int4_;
        unpacked_int4_data_ = std::move(other.unpacked_int4_data_);

        other.fd_ = -1;
        other.mapped_data_ = nullptr;
        other.file_size_ = 0;
        other.is_int4_ = false;
    }
    return *this;
}

const std::vector<size_t>& GraphFile::MappedFile::shape() const { 
    return shape_; 
}

Precision GraphFile::MappedFile::precision() const { 
    return precision_; 
}

size_t GraphFile::MappedFile::byte_size() const {
    return byte_size_;
}

const void* GraphFile::MappedFile::scales_data() const {
    return static_cast<const char*>(mapped_data_) + scales_offset_;
}


void GraphFile::MappedFile::unpack_int4_if_needed() const {
    if (!is_int4_ || unpacked_int4_data_) {
        return;  // Not INT4, or already unpacked
    }

    // Calculate unpacked size (2x packed size)
    size_t unpacked_count = byte_size_ * 2;
    unpacked_int4_data_ = std::make_unique<int8_t[]>(unpacked_count);

    // Get packed data from mmap
    const uint8_t* packed = reinterpret_cast<const uint8_t*>(
        static_cast<const char*>(mapped_data_) + data_offset_);

    // Unpack INT4 to INT8
    cactus_unpack_int4_to_int8(packed, unpacked_int4_data_.get(), unpacked_count);
}

void* GraphFile::MappedFile::data() {
    if (is_int4_) {
        unpack_int4_if_needed();
        return unpacked_int4_data_.get();
    }
    return static_cast<char*>(mapped_data_) + data_offset_;
}

const void* GraphFile::MappedFile::data() const {
    if (is_int4_) {
        unpack_int4_if_needed();
        return unpacked_int4_data_.get();
    }
    return static_cast<const char*>(mapped_data_) + data_offset_;
}

template<typename T>
const T* GraphFile::MappedFile::typed_data() const {
    return static_cast<const T*>(data());
}

GraphFile::LoadedNode GraphFile::MappedFile::load_into_graph(CactusGraph& graph) const {
    Precision eff_prec = effective_precision();

    size_t node_id = graph.input(shape_, eff_prec);
    graph.set_external_input(node_id, const_cast<void*>(data()), eff_prec);

    if ((eff_prec == Precision::INT8) && group_size_ > 0) {
        graph.set_grouped_scales(node_id, group_size_, num_groups_,
                                 const_cast<void*>(scales_data()));
    }

    size_t reported_byte_size = is_int4_ ? byte_size_ * 2 : byte_size_;
    return {node_id, shape_, eff_prec, reported_byte_size};
}

void GraphFile::MappedFile::parse_header() {
    const char* ptr = static_cast<const char*>(mapped_data_);
    size_t offset = 0;
    
    const size_t min_header_size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
    if (file_size_ < min_header_size) {
        throw std::runtime_error("File too small: insufficient data for header");
    }
    
    uint32_t ndim = *reinterpret_cast<const uint32_t*>(ptr + offset);
    offset += sizeof(uint32_t);
    
   const size_t dims_size = ndim * sizeof(uint64_t);
    if (offset + dims_size + sizeof(uint32_t) + sizeof(uint64_t) > file_size_) {
        throw std::runtime_error("File corrupted: insufficient data for header with " + std::to_string(ndim) + " dimensions");
    }
    
    shape_.resize(ndim);
    for (uint32_t i = 0; i < ndim; i++) {
        uint64_t dim_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
        shape_[i] = static_cast<size_t>(dim_val);
        offset += sizeof(uint64_t);
    }
    
    uint32_t prec_val = *reinterpret_cast<const uint32_t*>(ptr + offset);
    precision_ = static_cast<Precision>(prec_val);
    offset += sizeof(uint32_t);

    is_int4_ = (precision_ == Precision::INT4);

    uint64_t size_val = *reinterpret_cast<const uint64_t*>(ptr + offset);
    byte_size_ = static_cast<size_t>(size_val);
    offset += sizeof(uint64_t);

    if (precision_ == Precision::INT8 || precision_ == Precision::INT4) {
        if (offset + sizeof(uint32_t) + sizeof(uint64_t) > file_size_) {
            throw std::runtime_error("File corrupted: missing group-wise quantization parameters for quantized tensor");
        }
        group_size_ = *reinterpret_cast<const uint32_t*>(ptr + offset);
        offset += sizeof(uint32_t);

        num_groups_ = *reinterpret_cast<const uint64_t*>(ptr + offset);
        offset += sizeof(uint64_t);

        data_offset_ = offset;

        if (group_size_ > 0 && num_groups_ > 0) {
            scales_offset_ = data_offset_ + byte_size_;
            size_t N = shape_.size() >= 1 ? shape_[0] : 1;
            size_t scales_byte_size = N * num_groups_ * sizeof(__fp16);

            if (scales_offset_ + scales_byte_size > file_size_) {
                throw std::runtime_error("File corrupted: scales data extends beyond file size");
            }
        } else {
            scales_offset_ = 0;
        }
    } else {
        group_size_ = 0;
        num_groups_ = 0;
        scales_offset_ = 0;
        data_offset_ = offset;
        is_int4_ = false;
    }

    if (data_offset_ + byte_size_ > file_size_) {
        throw std::runtime_error("File corrupted: data extends beyond file size");
    }
}

template const int8_t* GraphFile::MappedFile::typed_data<int8_t>() const;
template const float* GraphFile::MappedFile::typed_data<float>() const;
template const uint16_t* GraphFile::MappedFile::typed_data<uint16_t>() const;
template const uint8_t* GraphFile::MappedFile::typed_data<uint8_t>() const;

GraphFile::MappedFile GraphFile::mmap_load(const std::string& filename) {
    return MappedFile(filename);
} 
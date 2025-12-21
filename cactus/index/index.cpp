#include "index.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <numeric>
#include <unordered_set>
#include <queue>

namespace cactus {
namespace index {

    float dot_product(const float *a, const float *b, size_t dim) {
        float result = 0.0f;
        for (size_t i = 0; i < dim; ++i) {
            result += a[i] * b[i];
        }
        return result;
    }

    void normalize(float *v, size_t dim) {
        float norm = std::sqrt(dot_product(v, v, dim));

        if (norm < 1e-6f) {
            return;
        }

        for (size_t i = 0; i < dim; ++i) {
            v[i] /= norm;
        }
    }

    Index::Index(const std::string& index_path, const std::string& data_path, uint32_t embedding_dim):
        index_path_(index_path), data_path_(data_path), embedding_dim_(embedding_dim) {
        index_fd_ = open(index_path.c_str(), O_RDWR);
        data_fd_ = open(data_path.c_str(), O_RDWR);

        struct stat index_st, data_st;
        if (fstat(index_fd_, &index_st)) {
            close(index_fd_);
            throw std::runtime_error("Cannot get index file size: " + index_path);
        }
        index_file_size_ = index_st.st_size;

        if (fstat(data_fd_, &data_st)) {
            close(data_fd_);
            throw std::runtime_error("Cannot get data file size: " + data_path);
        }
        data_file_size_ = data_st.st_size;

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd_, 0);
        if (mapped_index_ == MAP_FAILED) {
            close(index_fd_);
            throw std::runtime_error("Cannot map file: " + index_path);
        }

        close(index_fd_);
        index_fd_ = -1;

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, data_fd_, 0);
        if (mapped_data_ == MAP_FAILED) {
            close(data_fd_);
            throw std::runtime_error("Cannot map file: " + data_path);
        }

        close(data_fd_);
        data_fd_ = -1;

        parse_index_header();
        parse_data_header();

        build_doc_id_map();
    }

    void Index::add_documents(const std::vector<Document>& documents) {
        validate_documents(documents);

        int index_fd = open(index_path_.c_str(), O_RDWR);
        if (index_fd < 0) {
            throw std::runtime_error("Cannot open index file for writing: " + index_path_);
        }

        int data_fd = open(data_path_.c_str(), O_RDWR);
        if (data_fd < 0) {
            close(index_fd);
            throw std::runtime_error("Cannot open data file for writing: " + data_path_);
        }

        off_t data_offset = lseek(data_fd, 0, SEEK_END);

        std::vector<uint64_t> data_offsets;
        data_offsets.reserve(documents.size());

        std::vector<uint32_t> data_sizes;
        data_sizes.reserve(documents.size());

        for (const auto& doc : documents) {
            data_offsets.emplace_back(data_offset);

            const DataEntry entry {
                static_cast<uint8_t>(doc.id.size()),
                static_cast<uint16_t>(doc.content.size()),
                static_cast<uint16_t>(doc.metadata.size())
            };

            if (write(data_fd, &entry, sizeof(DataEntry)) != sizeof(DataEntry)
                || write(data_fd, doc.id.data(), doc.id.size()) != static_cast<ssize_t>(doc.id.size())
                || write(data_fd, doc.content.data(), doc.content.size()) != static_cast<ssize_t>(doc.content.size())
                || write(data_fd, doc.metadata.data(), doc.metadata.size()) != static_cast<ssize_t>(doc.metadata.size())) {
                close(index_fd);
                close(data_fd);
                throw std::runtime_error("Failed to write document entry");
            }

            uint32_t entry_size = sizeof(DataEntry) + doc.id.size() + doc.content.size() + doc.metadata.size();
            data_sizes.emplace_back(entry_size);
            data_offset += entry_size;
        }

        lseek(index_fd, 0, SEEK_END);

        for (size_t i = 0; i < documents.size(); ++i) {
            const auto& doc = documents[i];

            const IndexEntry entry {
                data_offsets[i],
                data_sizes[i],
                0
            };

            size_t embedding_bytes = embedding_dim_ * sizeof(float);
            if (write(index_fd, &entry, sizeof(IndexEntry)) != sizeof(IndexEntry)
                || write(index_fd, doc.embedding.data(), embedding_bytes) != static_cast<ssize_t>(embedding_bytes)) {
                close(index_fd);
                close(data_fd);
                throw std::runtime_error("Failed to write index entry");
            }

            doc_id_map_[doc.id] = num_documents_ + i;
        }

        munmap(mapped_index_, index_file_size_);
        munmap(mapped_data_, data_file_size_);

        index_file_size_ += documents.size() * IndexEntry::size(embedding_dim_);
        data_file_size_ += std::accumulate(data_sizes.begin(), data_sizes.end(), size_t{0});

        mapped_index_ = mmap(nullptr, index_file_size_, PROT_READ | PROT_WRITE, MAP_SHARED, index_fd, 0);
        if (mapped_index_ == MAP_FAILED) {
            close(index_fd);
            close(data_fd);
            throw std::runtime_error("Failed to remap index file");
        }

        mapped_data_ = mmap(nullptr, data_file_size_, PROT_READ, MAP_PRIVATE, data_fd, 0);
        if (mapped_data_ == MAP_FAILED) {
            munmap(mapped_index_, index_file_size_);
            close(index_fd);
            close(data_fd);
            throw std::runtime_error("Failed to remap data file");
        }

        close(index_fd);
        close(data_fd);

        num_documents_ += documents.size();
        IndexHeader* header = reinterpret_cast<IndexHeader*>(mapped_index_);
        header->num_documents = num_documents_;
    }

    void Index::delete_documents(const std::vector<std::string>& doc_ids) {
        validate_doc_ids(doc_ids);

        char* index_ptr = static_cast<char*>(mapped_index_);
        char* entries = index_ptr + sizeof(IndexHeader);

        for (const auto& doc_id: doc_ids) {
            uint32_t i = doc_id_map_.at(doc_id);
            IndexEntry& entry = *reinterpret_cast<IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));

            entry.flags |= 0x1;
            doc_id_map_.erase(doc_id);
        }
    }

    std::vector<Document> Index::get_documents(const std::vector<std::string>& doc_ids) {
        validate_doc_ids(doc_ids);

        std::vector<Document> results;
        results.reserve(doc_ids.size());

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);
        const char* data_ptr = static_cast<const char*>(mapped_data_);

        for (const auto& doc_id: doc_ids) {
            uint32_t i = doc_id_map_.at(doc_id);
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));
            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);

            const float* entry_embedding = entry.embedding();
            results.emplace_back(
                doc_id,
                std::vector<float>(entry_embedding, entry_embedding + embedding_dim_),
                std::string(data_entry->content(), data_entry->content_len),
                std::string(data_entry->metadata(), data_entry->metadata_len)
            );
        }

        return results;
    }

    std::vector<SearchResult> Index::query(const std::vector<float>& embedding, const SearchOptions& options) {
        if (options.top_k == 0) {
            return {};
        }

        std::vector<float> normalized_embedding(embedding);
        normalize(normalized_embedding.data(), embedding_dim_);

        auto cmp = [](const SearchResult& a, const SearchResult& b) {
            return a.score > b.score; 
        };
        std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(cmp)> top_results(cmp);

        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);
        const char* data_ptr = static_cast<const char*>(mapped_data_);

        for (uint32_t i = 0; i < num_documents_; ++i) {
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));
            const float* entry_embedding = entry.embedding();

            if (entry.flags & 0x1) {
                continue;
            }

            float score = dot_product(normalized_embedding.data(), entry_embedding, embedding_dim_);

            if (score < options.score_threshold) {
                continue;
            }

            if (top_results.size() < options.top_k) {
                const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
                top_results.emplace(std::string(data_entry->doc_id(), data_entry->doc_id_len), score);
            } else if (score > top_results.top().score) {
                top_results.pop();
                const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
                top_results.emplace(std::string(data_entry->doc_id(), data_entry->doc_id_len), score);
            }
        }

        std::vector<SearchResult> results(top_results.size());
        for (size_t i = top_results.size(); i-- > 0;) {
            results[i] = top_results.top();
            top_results.pop();
        }

        return results;
    }

    void Index::parse_index_header() {
        const char* index_ptr = static_cast<const char*>(mapped_index_);
        size_t offset = 0;

        if (index_file_size_ < sizeof(IndexHeader)) {
            throw std::runtime_error("Index file too small: insufficient data for header");
        }

        IndexHeader header;
        header.magic = *reinterpret_cast<const decltype(header.magic)*>(index_ptr + offset);
        offset += sizeof(header.magic);

        if (header.magic != MAGIC) {
            throw std::runtime_error("Invalid index file magic number");
        }

        header.version = *reinterpret_cast<const decltype(header.version)*>(index_ptr + offset);
        offset += sizeof(header.version);

        if (header.version != VERSION) {
            throw std::runtime_error("Index file version mismatch");
        }

        header.embedding_dim = *reinterpret_cast<const decltype(header.embedding_dim)*>(index_ptr + offset);
        offset += sizeof(header.embedding_dim);

        if (header.embedding_dim != embedding_dim_) {
            throw std::runtime_error("Embedding dimension mismatch");
        }

        num_documents_ = *reinterpret_cast<const decltype(header.num_documents)*>(index_ptr + offset);
    }

    void Index::parse_data_header() {
        const char* data_ptr = static_cast<const char*>(mapped_data_);
        size_t offset = 0;

        if (data_file_size_ < sizeof(DataHeader)) {
            throw std::runtime_error("Data file too small: insufficient data for header");
        }

        DataHeader header;
        header.magic = *reinterpret_cast<const decltype(header.magic)*>(data_ptr + offset);
        offset += sizeof(header.magic);

        if (header.magic != MAGIC) {
            throw std::runtime_error("Invalid data file magic number");
        }

        header.version = *reinterpret_cast<const decltype(header.version)*>(data_ptr + offset);
        offset += sizeof(header.version);

        if (header.version != VERSION) {
            throw std::runtime_error("Data file version mismatch");
        }
    }

    void Index::build_doc_id_map() {
        const char* index_ptr = static_cast<const char*>(mapped_index_);
        const char* entries = index_ptr + sizeof(IndexHeader);
        const char* data_ptr = static_cast<const char*>(mapped_data_);

        for (uint32_t i = 0; i < num_documents_; ++i) {
            const IndexEntry& entry = *reinterpret_cast<const IndexEntry*>(entries + i * IndexEntry::size(embedding_dim_));

            if (entry.flags & 0x1) {
                continue;
            }

            const DataEntry* data_entry = reinterpret_cast<const DataEntry*>(data_ptr + entry.data_offset);
            const std::string doc_id = std::string(data_entry->doc_id(), data_entry->doc_id_len);

            doc_id_map_[doc_id] = i;
        }
    }

    void Index::validate_documents(const std::vector<Document>& documents) {
        if (documents.empty()) {
            throw std::runtime_error("Documents vector is empty");
        }

        std::unordered_set<std::string> seen_ids;
        for (const auto& doc : documents) {
            if (doc.id.empty()) {
                throw std::runtime_error("Document ID cannot be empty");
            }

            if (doc.id.size() > 255) {
                throw std::runtime_error("Document ID too long (max 255 bytes): " + doc.id);
            }

            if (doc.content.size() > 65535) {
                throw std::runtime_error("Document content too long (max 65535 bytes) for ID: " + doc.id);
            }

            if (doc.metadata.size() > 65535) {
                throw std::runtime_error("Document metadata too long (max 65535 bytes) for ID: " + doc.id);
            }

            if (doc.embedding.size() != embedding_dim_) {
                throw std::runtime_error("Document embedding dimension mismatch for ID: " + doc.id);
            }

            if (seen_ids.find(doc.id) != seen_ids.end()) {
                throw std::runtime_error("Duplicate document ID in input: " + doc.id);
            }
            seen_ids.insert(doc.id);

            if (doc_id_map_.find(doc.id) != doc_id_map_.end()) {
                throw std::runtime_error("Document ID already exists: " + doc.id);
            }
        }
    }

    void Index::validate_doc_ids(const std::vector<std::string>& doc_ids) {
        if (doc_ids.empty()) {
            throw std::runtime_error("Document IDs vector is empty");
        }

        std::unordered_set<std::string> seen_ids;
        for (const auto& doc_id : doc_ids) {
            if (doc_id.empty()) {
                throw std::runtime_error("Document ID cannot be empty");
            }

            if (seen_ids.find(doc_id) != seen_ids.end()) {
                throw std::runtime_error("Duplicate document ID in input: " + doc_id);
            }
            seen_ids.insert(doc_id);

            if (doc_id_map_.find(doc_id) == doc_id_map_.end()) {
                throw std::runtime_error("Document ID not found: " + doc_id);
            }
        }
    }

} // namespace index
} // namespace cactus
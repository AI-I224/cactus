#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace cactus {
namespace index {

    constexpr uint32_t MAGIC = 0x43414354;
    constexpr uint32_t VERSION = 1;

    struct Document {
        int id;
        std::vector<float> embedding;
        std::string content;
        std::string metadata;
    };

    struct SearchResult {
        int doc_id;
        float score;
    };

    struct SearchOptions {
        size_t top_k = 10;
        float score_threshold = -1.0f;
    };

    class Index {
        public:
            Index(const std::string& index_path, const std::string& data_path, size_t embedding_dim);
            ~Index();

            Index(const Index&) = delete;
            Index& operator=(const Index&) = delete;
            Index(Index&&) = delete;
            Index& operator=(Index&&) = delete;

            void add_documents(const std::vector<Document>& documents);
            void delete_documents(const std::vector<int>& doc_ids);
            std::vector<Document> get_documents(const std::vector<int>& doc_ids);
            std::vector<std::vector<SearchResult>> query(const std::vector<std::vector<float>>& embeddings, const SearchOptions& options);
            void compact();

        private:
            struct IndexHeader {
                uint32_t magic;
                uint32_t version;
                uint32_t embedding_dim;
                uint32_t num_documents;
            };

            struct IndexEntry {
                int32_t doc_id;
                uint64_t data_offset;
                uint8_t flags; // bit 0: tombstone

                const __fp16* embedding() const {
                    return reinterpret_cast<const __fp16*>(this + 1);
                }

                static size_t size(size_t embedding_dim) {
                    return sizeof(IndexEntry) + embedding_dim * sizeof(__fp16);
                }
            };

            struct DataHeader {
                uint32_t magic;
                uint32_t version;
            };

            struct DataEntry {
                uint16_t content_len;
                uint16_t metadata_len;

                const char* content() const {
                    return reinterpret_cast<const char*>(this + 1);
                }

                const char* metadata() const {
                    return content() + content_len;
                }
            };

            void parse_index_header();
            void parse_data_header();
            void build_doc_id_map();
            void validate_documents(const std::vector<Document>& documents);
            void validate_doc_ids(const std::vector<int>& doc_ids);
            ssize_t write_full(int fd, const void* buf, size_t count);

            std::unordered_map<int, uint32_t> doc_id_map_;

            std::string index_path_, data_path_;
            size_t embedding_dim_;
            size_t index_entry_size_;
            uint32_t num_documents_;

            int index_fd_, data_fd_;
            void *mapped_index_, *mapped_data_;
            size_t index_file_size_, data_file_size_;
    };

} // namespace index
} // namespace cactus
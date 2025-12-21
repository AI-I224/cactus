#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace cactus {
namespace index {

    constexpr uint32_t MAGIC = 0x43414354;
    constexpr uint32_t VERSION = 1;

    struct Document {
        std::string id;
        std::vector<float> embedding;
        std::string content;
        std::string metadata;
    };

    struct SearchResult {
        std::string doc_id;
        float score;
    };

    struct SearchOptions {
        uint32_t top_k = 10;
        float score_threshold = 0.0f;
    };

    class Index {
        public:
            Index(const std::string& index_path, const std::string& data_path, uint32_t embedding_dim);

            void add_documents(const std::vector<Document>& documents);
            void delete_documents(const std::vector<std::string>& doc_ids);
            std::vector<Document> get_documents(const std::vector<std::string>& doc_ids);
            std::vector<std::vector<SearchResult>> query(const std::vector<std::vector<float>>& embeddings, const SearchOptions& options);

        private:
            struct IndexHeader {
                uint32_t magic;
                uint32_t version;
                uint32_t embedding_dim;
                uint32_t num_documents;
            };

            struct IndexEntry {
                uint64_t data_offset;
                uint32_t data_size;
                uint8_t flags; // bit 0: tombstone

                const float* embedding() const {
                    return reinterpret_cast<const float*>(this + 1);
                }

                static size_t size(uint32_t embedding_dim) {
                    return sizeof(IndexEntry) + embedding_dim * sizeof(float);
                }
            };

            struct DataHeader {
                uint32_t magic;
                uint32_t version;
            };

            struct DataEntry {
                uint8_t doc_id_len;
                uint16_t content_len;
                uint16_t metadata_len;

                const char* doc_id() const {
                    return reinterpret_cast<const char*>(this + 1);
                }

                const char* content() const {
                    return doc_id() + doc_id_len;
                }

                const char* metadata() const {
                    return content() + content_len;
                }

                size_t total_size() const {
                    return sizeof(DataEntry) + doc_id_len + content_len + metadata_len;
                }
            };

            void parse_index_header();
            void parse_data_header();
            void build_doc_id_map();
            void validate_documents(const std::vector<Document>& documents);
            void validate_doc_ids(const std::vector<std::string>& doc_ids);

            std::unordered_map<std::string, uint32_t> doc_id_map_;

            std::string index_path_, data_path_;
            uint32_t embedding_dim_;
            uint32_t num_documents_;

            int index_fd_, data_fd_;
            void *mapped_index_, *mapped_data_;
            size_t index_file_size_, data_file_size_;
    };

} // namespace index
} // namespace cactus
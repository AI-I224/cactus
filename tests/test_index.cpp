#include "../cactus/ffi/cactus_ffi.h"
#include "test_utils.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <limits>

const char* g_index_path = std::getenv("CACTUS_INDEX_PATH");

std::vector<float> random_embedding(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> embedding(dim);
    for (auto& val : embedding) {
        val = dist(gen);
    }
    return embedding;
}

void cleanup_test_dir(const std::string& dir_path) {
    std::string index_file = dir_path + "/index.bin";
    std::string data_file = dir_path + "/data.bin";
    std::string backup_index = dir_path + "/index.bin.backup";
    std::string backup_data = dir_path + "/data.bin.backup";
    unlink(index_file.c_str());
    unlink(data_file.c_str());
    unlink(backup_index.c_str());
    unlink(backup_data.c_str());
    rmdir(dir_path.c_str());
}

void create_test_dir(const std::string& dir_path) {
    mkdir(dir_path.c_str(), 0755);
}

// ============================================================================
// Constructor Tests
// ============================================================================

bool test_constructor_valid() {
    const std::string dir_path = std::string(g_index_path) + "/test_constructor_valid";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    bool success = (index != nullptr);

    if (index) cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_constructor_missing_index() {
    const std::string dir_path = std::string(g_index_path) + "/test_missing";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    std::ofstream data_file(dir_path + "/data.bin", std::ios::binary);
    data_file.close();

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    bool success = (index == nullptr);

    cleanup_test_dir(dir_path);
    return success;
}

bool test_constructor_missing_data() {
    const std::string dir_path = std::string(g_index_path) + "/test_missing_data";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    std::ofstream index_file(dir_path + "/index.bin", std::ios::binary);
    index_file.close();

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    bool success = (index == nullptr);

    cleanup_test_dir(dir_path);
    return success;
}

bool test_constructor_wrong_magic() {
    const std::string dir_path = std::string(g_index_path) + "/test_wrong_magic";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    std::ofstream index_file(dir_path + "/index.bin", std::ios::binary);
    uint32_t wrong_magic = 0xDEADBEEF;
    index_file.write(reinterpret_cast<const char*>(&wrong_magic), sizeof(uint32_t));
    index_file.close();

    std::ofstream data_file(dir_path + "/data.bin", std::ios::binary);
    data_file.close();

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    bool success = (index == nullptr);

    cleanup_test_dir(dir_path);
    return success;
}

bool test_constructor_dimension_mismatch() {
    const std::string dir_path = std::string(g_index_path) + "/test_dim_mismatch";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index1 = cactus_index_init(dir_path.c_str(), 1024);
    cactus_index_destroy(index1);

    cactus_index_t index2 = cactus_index_init(dir_path.c_str(), 256);
    bool success = (index2 == nullptr);

    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Add Document Tests
// ============================================================================

bool test_add_document() {
    const std::string dir_path = std::string(g_index_path) + "/test_add_document";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs[] = {"content1"};
    const char* metas[] = {"metadata1"};
    auto emb = random_embedding(1024);
    const float* embeddings[] = {emb.data()};

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);

    char* doc_buffers[1];
    doc_buffers[0] = (char*)malloc(65536);
    size_t doc_sizes[1] = {65536};

    int get_result = cactus_index_get(index, ids, 1, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0) && (get_result == 0);

    free(doc_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_add_multiple_documents() {
    const std::string dir_path = std::string(g_index_path) + "/test_add_multiple";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs.push_back("content" + std::to_string(i));
        meta_strs.push_back("metadata" + std::to_string(i));
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    int get_ids[] = {0, 5, 9};
    char* doc_buffers[3];
    for (int i = 0; i < 3; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[3] = {65536, 65536, 65536};

    int get_result = cactus_index_get(index, get_ids, 3, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0) && (get_result == 0);

    for (int i = 0; i < 3; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_add_with_null_metadata() {
    const std::string dir_path = std::string(g_index_path) + "/test_null_metadata";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1, 2};
    const char* docs[] = {"doc1", "doc2"};
    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);
    const float* embeddings[] = {emb1.data(), emb2.data()};

    int result = cactus_index_add(index, ids, docs, nullptr, embeddings, 2, 1024);

    bool success = (result == 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_add_after_delete() {
    const std::string dir_path = std::string(g_index_path) + "/test_add_after_delete";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs1[] = {"first"};
    const char* metas[] = {"meta1"};
    auto emb1 = random_embedding(1024);
    const float* embeddings1[] = {emb1.data()};

    cactus_index_add(index, ids, docs1, metas, embeddings1, 1, 1024);
    cactus_index_delete(index, ids, 1);

    const char* docs2[] = {"second"};
    auto emb2 = random_embedding(1024);
    const float* embeddings2[] = {emb2.data()};
    int add_result = cactus_index_add(index, ids, docs2, metas, embeddings2, 1, 1024);

    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;
    int get_result = cactus_index_get(index, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);

    bool success = (add_result == 0) && (get_result == 0) && (strcmp(doc_buffer, "second") == 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Get Document Tests
// ============================================================================

bool test_get_document() {
    const std::string dir_path = std::string(g_index_path) + "/test_get_document";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs[] = {"test content"};
    const char* metas[] = {"test metadata"};
    auto emb = random_embedding(1024);
    const float* embeddings[] = {emb.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);

    char* doc_buffers[1];
    char* meta_buffers[1];
    float* emb_buffers[1];
    doc_buffers[0] = (char*)malloc(65536);
    meta_buffers[0] = (char*)malloc(65536);
    emb_buffers[0] = (float*)malloc(1024 * sizeof(float));

    size_t doc_sizes[1] = {65536};
    size_t meta_sizes[1] = {65536};
    size_t emb_sizes[1] = {1024};

    int result = cactus_index_get(index, ids, 1, doc_buffers, doc_sizes, meta_buffers, meta_sizes, emb_buffers, emb_sizes);

    bool success = (result == 0) &&
                   (strcmp(doc_buffers[0], "test content") == 0) &&
                   (strcmp(meta_buffers[0], "test metadata") == 0) &&
                   (emb_sizes[0] == 1024);

    free(doc_buffers[0]);
    free(meta_buffers[0]);
    free(emb_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_get_multiple_documents() {
    const std::string dir_path = std::string(g_index_path) + "/test_get_multiple";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs(10), meta_strs(10);

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs[i] = "content" + std::to_string(i);
        meta_strs[i] = "meta" + std::to_string(i);
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    int get_ids[] = {2, 5, 8};
    char* doc_buffers[3];
    for (int i = 0; i < 3; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[3] = {65536, 65536, 65536};

    int result = cactus_index_get(index, get_ids, 3, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0) &&
                   (strcmp(doc_buffers[0], "content2") == 0) &&
                   (strcmp(doc_buffers[1], "content5") == 0) &&
                   (strcmp(doc_buffers[2], "content8") == 0);

    for (int i = 0; i < 3; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_get_only_documents() {
    const std::string dir_path = std::string(g_index_path) + "/test_get_only_docs";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1, 2, 3};
    const char* docs[] = {"doc1", "doc2", "doc3"};
    const char* metas[] = {"meta1", "meta2", "meta3"};
    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);
    auto emb3 = random_embedding(1024);
    const float* embeddings[] = {emb1.data(), emb2.data(), emb3.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 3, 1024);

    char* doc_buffers[3];
    for (int i = 0; i < 3; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[3] = {65536, 65536, 65536};

    int result = cactus_index_get(index, ids, 3, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0) &&
                   (strcmp(doc_buffers[0], "doc1") == 0) &&
                   (strcmp(doc_buffers[1], "doc2") == 0) &&
                   (strcmp(doc_buffers[2], "doc3") == 0);

    for (int i = 0; i < 3; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_get_after_compact() {
    const std::string dir_path = std::string(g_index_path) + "/test_get_after_compact";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs(10), meta_strs(10);

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs[i] = "content" + std::to_string(i);
        meta_strs[i] = "meta" + std::to_string(i);
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    int delete_ids[] = {1, 3, 5, 7, 9};
    cactus_index_delete(index, delete_ids, 5);
    cactus_index_compact(index);

    int get_ids[] = {0, 2, 4, 6, 8};
    char* doc_buffers[5];
    for (int i = 0; i < 5; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[5] = {65536, 65536, 65536, 65536, 65536};

    int result = cactus_index_get(index, get_ids, 5, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0) &&
                   (strcmp(doc_buffers[0], "content0") == 0) &&
                   (strcmp(doc_buffers[1], "content2") == 0) &&
                   (strcmp(doc_buffers[2], "content4") == 0) &&
                   (strcmp(doc_buffers[3], "content6") == 0) &&
                   (strcmp(doc_buffers[4], "content8") == 0);

    for (int i = 0; i < 5; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Delete Tests
// ============================================================================

bool test_delete_document() {
    const std::string dir_path = std::string(g_index_path) + "/test_delete_document";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1, 2};
    const char* docs[] = {"content1", "content2"};
    const char* metas[] = {"meta1", "meta2"};
    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);
    const float* embeddings[] = {emb1.data(), emb2.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 2, 1024);

    int delete_ids[] = {1};
    int del_result = cactus_index_delete(index, delete_ids, 1);

    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;
    int get_result = cactus_index_get(index, delete_ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);

    bool success = (del_result == 0) && (get_result != 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_delete_alternating() {
    const std::string dir_path = std::string(g_index_path) + "/test_delete_alternating";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int i = 0; i < 20; ++i) {
        int id = i;
        const char* doc = "doc";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    }

    int delete_ids[10];
    for (int i = 0; i < 10; ++i) delete_ids[i] = i * 2;
    cactus_index_delete(index, delete_ids, 10);

    bool all_deleted = true;
    for (int id : {0, 2, 4, 6}) {
        char* doc_buffer = (char*)malloc(65536);
        size_t doc_size = 65536;
        int result = cactus_index_get(index, &id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
        if (result == 0) all_deleted = false;
        free(doc_buffer);
    }

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return all_deleted;
}

bool test_delete_then_query() {
    const std::string dir_path = std::string(g_index_path) + "/test_delete_then_query";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int i = 0; i < 10; ++i) {
        int id = i;
        const char* doc = "doc";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    }

    int delete_ids[] = {0, 1, 2};
    cactus_index_delete(index, delete_ids, 3);

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":10}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] <= 7);
    for (size_t i = 0; i < id_sizes[0]; ++i) {
        if (id_buffers[0][i] == 0 || id_buffers[0][i] == 1 || id_buffers[0][i] == 2) {
            success = false;
            break;
        }
    }

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Compact Tests
// ============================================================================

bool test_compact_reclaim_space() {
    const std::string dir_path = std::string(g_index_path) + "/test_compact_reclaim";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs.push_back("content" + std::to_string(i));
        meta_strs.push_back("metadata" + std::to_string(i));
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    std::string index_file = dir_path + "/index.bin";
    struct stat st_before;
    stat(index_file.c_str(), &st_before);
    size_t index_size_before = st_before.st_size;

    int delete_ids[] = {0, 2, 4, 6, 8};
    cactus_index_delete(index, delete_ids, 5);
    int compact_result = cactus_index_compact(index);

    struct stat st_after;
    stat(index_file.c_str(), &st_after);
    size_t index_size_after = st_after.st_size;

    int get_ids[] = {1, 3, 5, 7, 9};
    char* doc_buffers[5];
    for (int i = 0; i < 5; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[5] = {65536, 65536, 65536, 65536, 65536};

    int get_result = cactus_index_get(index, get_ids, 5, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (compact_result == 0) && (get_result == 0) && (index_size_after < index_size_before);

    for (int i = 0; i < 5; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_compact_query_after() {
    const std::string dir_path = std::string(g_index_path) + "/test_compact_query";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);
    auto emb3 = random_embedding(1024);

    int ids[] = {1, 2, 3};
    const char* docs[] = {"content1", "content2", "content3"};
    const char* metas[] = {"meta1", "meta2", "meta3"};
    const float* embeddings[] = {emb1.data(), emb2.data(), emb3.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 3, 1024);

    int delete_ids[] = {2};
    cactus_index_delete(index, delete_ids, 1);
    cactus_index_compact(index);

    const float* queries[] = {emb1.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(2 * sizeof(int));
    score_buffers[0] = (float*)malloc(2 * sizeof(float));
    size_t id_sizes[1] = {2};
    size_t score_sizes[1] = {2};

    const char* options = "{\"top_k\":2}";
    int query_result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (query_result == 0) && (id_sizes[0] >= 1) && (id_buffers[0][0] == 1);

    int get_ids[] = {1};
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;
    int get_result = cactus_index_get(index, get_ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);

    success = success && (get_result == 0) && (strcmp(doc_buffer, "content1") == 0);

    free(doc_buffer);
    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_compact_empty_index() {
    const std::string dir_path = std::string(g_index_path) + "/test_compact_empty";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int result = cactus_index_compact(index);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return (result == 0);
}

bool test_compact_all_deleted() {
    const std::string dir_path = std::string(g_index_path) + "/test_compact_all_deleted";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[5];
    const char* docs[5];
    const char* metas[5];
    std::vector<std::vector<float>> embeddings_data(5);
    const float* embeddings[5];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 5; ++i) {
        ids[i] = i;
        doc_strs.push_back("content");
        meta_strs.push_back("meta");
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 5, 1024);
    cactus_index_delete(index, ids, 5);
    cactus_index_compact(index);

    std::string index_file = dir_path + "/index.bin";
    struct stat st;
    stat(index_file.c_str(), &st);
    size_t index_size = st.st_size;
    size_t header_size = 16;

    bool success = (index_size == header_size);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_compact_large_gaps() {
    const std::string dir_path = std::string(g_index_path) + "/test_compact_gaps";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int i = 0; i < 100; ++i) {
        int id = i;
        const char* doc = "doc";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    }

    for (int i = 10; i < 90; ++i) {
        cactus_index_delete(index, &i, 1);
    }

    cactus_index_compact(index);

    int get_ids[] = {0, 5, 90, 95};
    char* doc_buffers[4];
    for (int i = 0; i < 4; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[4] = {65536, 65536, 65536, 65536};

    int result = cactus_index_get(index, get_ids, 4, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);

    bool success = (result == 0);

    for (int i = 0; i < 4; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Query Tests
// ============================================================================

bool test_query_similarity() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_similarity";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);

    int ids[] = {1, 2};
    const char* docs[] = {"content1", "content2"};
    const char* metas[] = {"meta1", "meta2"};
    const float* embeddings[] = {emb1.data(), emb2.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 2, 1024);

    const float* queries[] = {emb1.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":1}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] >= 1) && (id_buffers[0][0] == 1);

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_topk() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_topk";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs.push_back("content");
        meta_strs.push_back("meta");
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":5}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] <= 5) && (id_sizes[0] > 0);

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_exact_match() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_exact";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    auto emb_exact = random_embedding(1024);
    int id = 1;
    const char* doc = "exact";
    const char* meta = "meta";
    const float* embedding = emb_exact.data();
    cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);

    for (int i = 2; i < 10; ++i) {
        int id_other = i;
        const char* doc_other = "other";
        auto emb_other = random_embedding(1024);
        const float* embedding_other = emb_other.data();
        cactus_index_add(index, &id_other, &doc_other, &meta, &embedding_other, 1, 1024);
    }

    const float* queries[] = {emb_exact.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(1 * sizeof(int));
    score_buffers[0] = (float*)malloc(1 * sizeof(float));
    size_t id_sizes[1] = {1};
    size_t score_sizes[1] = {1};

    const char* options = "{\"top_k\":1}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] == 1) && (id_buffers[0][0] == 1);

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_score_range() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_score_range";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int i = 0; i < 10; ++i) {
        int id = i;
        const char* doc = "doc";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    }

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":10}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] > 0);
    for (size_t i = 0; i < id_sizes[0]; ++i) {
        if (score_buffers[0][i] < -1.0f || score_buffers[0][i] > 1.0f) {
            success = false;
            break;
        }
    }

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_score_ordering() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_ordering";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[20];
    const char* docs[20];
    const char* metas[20];
    std::vector<std::vector<float>> embeddings_data(20);
    const float* embeddings[20];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 20; ++i) {
        ids[i] = i + 1;
        doc_strs.push_back("doc");
        meta_strs.push_back("meta");
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 20, 1024);

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":10}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0);
    for (size_t i = 1; i < id_sizes[0]; ++i) {
        if (score_buffers[0][i-1] < score_buffers[0][i]) {
            success = false;
            break;
        }
    }

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_score_threshold() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_threshold";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    auto emb_exact = random_embedding(1024);
    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data;
    const float* embeddings[10];
    std::vector<std::string> doc_strs, meta_strs;

    embeddings_data.push_back(emb_exact);
    ids[0] = 1;
    doc_strs.push_back("exact");
    meta_strs.push_back("meta");
    docs[0] = doc_strs[0].c_str();
    metas[0] = meta_strs[0].c_str();
    embeddings[0] = embeddings_data[0].data();

    for (int i = 1; i < 10; ++i) {
        ids[i] = i + 1;
        embeddings_data.push_back(random_embedding(1024));
        doc_strs.push_back("other");
        meta_strs.push_back("meta");
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    const float* queries[] = {emb_exact.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":10,\"score_threshold\":0.95}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] > 0);
    for (size_t i = 0; i < id_sizes[0]; ++i) {
        if (score_buffers[0][i] < 0.95f) {
            success = false;
            break;
        }
    }

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_threshold_default() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_threshold_default";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int i = 0; i < 5; ++i) {
        int id = i;
        const char* doc = "doc";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    }

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};

    int* id_buffers1[1];
    float* score_buffers1[1];
    id_buffers1[0] = (int*)malloc(5 * sizeof(int));
    score_buffers1[0] = (float*)malloc(5 * sizeof(float));
    size_t id_sizes1[1] = {5};
    size_t score_sizes1[1] = {5};

    const char* options1 = "{\"top_k\":5}";
    cactus_index_query(index, queries, 1, 1024, options1, id_buffers1, id_sizes1, score_buffers1, score_sizes1);

    int* id_buffers2[1];
    float* score_buffers2[1];
    id_buffers2[0] = (int*)malloc(5 * sizeof(int));
    score_buffers2[0] = (float*)malloc(5 * sizeof(float));
    size_t id_sizes2[1] = {5};
    size_t score_sizes2[1] = {5};

    const char* options2 = "{\"top_k\":5,\"score_threshold\":-1.0}";
    cactus_index_query(index, queries, 1, 1024, options2, id_buffers2, id_sizes2, score_buffers2, score_sizes2);

    bool success = (id_sizes1[0] == id_sizes2[0]);

    free(id_buffers1[0]);
    free(score_buffers1[0]);
    free(id_buffers2[0]);
    free(score_buffers2[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_empty_embeddings() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_empty_embeddings";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int id = 1;
    const char* doc = "doc";
    const char* meta = "meta";
    auto emb = random_embedding(1024);
    const float* embedding = emb.data();
    cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);

    int* id_buffers[1];
    float* score_buffers[1];
    size_t id_sizes[1];
    size_t score_sizes[1];

    int result = cactus_index_query(index, nullptr, 0, 1024, nullptr, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_query_batch() {
    const std::string dir_path = std::string(g_index_path) + "/test_query_batch";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[10];
    const char* docs[10];
    const char* metas[10];
    std::vector<std::vector<float>> embeddings_data(10);
    const float* embeddings[10];
    std::vector<std::string> doc_strs, meta_strs;

    for (int i = 0; i < 10; ++i) {
        ids[i] = i;
        doc_strs.push_back("content");
        meta_strs.push_back("meta");
        docs[i] = doc_strs[i].c_str();
        metas[i] = meta_strs[i].c_str();
        embeddings_data[i] = random_embedding(1024);
        embeddings[i] = embeddings_data[i].data();
    }

    cactus_index_add(index, ids, docs, metas, embeddings, 10, 1024);

    std::vector<std::vector<float>> query_data(5);
    const float* queries[5];
    for (int i = 0; i < 5; ++i) {
        query_data[i] = random_embedding(1024);
        queries[i] = query_data[i].data();
    }

    int* id_buffers[5];
    float* score_buffers[5];
    size_t id_sizes[5];
    size_t score_sizes[5];
    for (int i = 0; i < 5; ++i) {
        id_buffers[i] = (int*)malloc(3 * sizeof(int));
        score_buffers[i] = (float*)malloc(3 * sizeof(float));
        id_sizes[i] = 3;
        score_sizes[i] = 3;
    }

    const char* options = "{\"top_k\":3}";
    int result = cactus_index_query(index, queries, 5, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0);
    for (int i = 0; i < 5; ++i) {
        if (id_sizes[i] > 3 || id_sizes[i] == 0) {
            success = false;
        }
    }

    for (int i = 0; i < 5; ++i) {
        free(id_buffers[i]);
        free(score_buffers[i]);
    }
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Persistence Tests
// ============================================================================

bool test_persist_after_add() {
    const std::string dir_path = std::string(g_index_path) + "/test_persist_add";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        int ids[] = {1};
        const char* docs[] = {"persisted"};
        const char* metas[] = {"meta"};
        auto emb = random_embedding(1024);
        const float* embeddings[] = {emb.data()};
        cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
        cactus_index_destroy(index);
    }

    cactus_index_t index2 = cactus_index_init(dir_path.c_str(), 1024);
    int ids[] = {1};
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index2, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0) && (strcmp(doc_buffer, "persisted") == 0);

    free(doc_buffer);
    cactus_index_destroy(index2);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_persist_after_delete() {
    const std::string dir_path = std::string(g_index_path) + "/test_persist_delete";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        int ids[] = {1};
        const char* docs[] = {"doc"};
        const char* metas[] = {"meta"};
        auto emb = random_embedding(1024);
        const float* embeddings[] = {emb.data()};
        cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
        cactus_index_delete(index, ids, 1);
        cactus_index_destroy(index);
    }

    cactus_index_t index2 = cactus_index_init(dir_path.c_str(), 1024);
    int ids[] = {1};
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index2, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result != 0);

    free(doc_buffer);
    cactus_index_destroy(index2);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_persist_after_compact() {
    const std::string dir_path = std::string(g_index_path) + "/test_persist_compact";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        for (int i = 0; i < 10; ++i) {
            int id = i;
            const char* doc = "doc";
            const char* meta = "meta";
            auto emb = random_embedding(1024);
            const float* embedding = emb.data();
            cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
        }
        int delete_ids[] = {0, 1, 2};
        cactus_index_delete(index, delete_ids, 3);
        cactus_index_compact(index);
        cactus_index_destroy(index);
    }

    cactus_index_t index2 = cactus_index_init(dir_path.c_str(), 1024);
    int ids[] = {3};
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index2, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0);

    free(doc_buffer);
    cactus_index_destroy(index2);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_persist_reload_sequence() {
    const std::string dir_path = std::string(g_index_path) + "/test_persist_sequence";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        int id = 1;
        const char* doc = "doc1";
        const char* meta = "meta";
        auto emb = random_embedding(1024);
        const float* embedding = emb.data();
        cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
        cactus_index_destroy(index);
    }

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        int id = 1;
        char* doc_buffer = (char*)malloc(65536);
        size_t doc_size = 65536;
        int r1 = cactus_index_get(index, &id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
        free(doc_buffer);
        if (r1 != 0) {
            cactus_index_destroy(index);
            cleanup_test_dir(dir_path);
            return false;
        }

        int id2 = 2;
        const char* doc2 = "doc2";
        const char* meta = "meta";
        auto emb2 = random_embedding(1024);
        const float* embedding2 = emb2.data();
        cactus_index_add(index, &id2, &doc2, &meta, &embedding2, 1, 1024);
        cactus_index_destroy(index);
    }

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        int id = 2;
        char* doc_buffer = (char*)malloc(65536);
        size_t doc_size = 65536;
        int r2 = cactus_index_get(index, &id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
        free(doc_buffer);
        if (r2 != 0) {
            cactus_index_destroy(index);
            cleanup_test_dir(dir_path);
            return false;
        }

        int delete_id = 1;
        cactus_index_delete(index, &delete_id, 1);
        cactus_index_destroy(index);
    }

    {
        cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
        cactus_index_compact(index);
        cactus_index_destroy(index);
    }

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    int id2 = 2;
    char* doc_buffer2 = (char*)malloc(65536);
    size_t doc_size2 = 65536;
    int r2_final = cactus_index_get(index, &id2, 1, &doc_buffer2, &doc_size2, nullptr, nullptr, nullptr, nullptr);

    int id1 = 1;
    char* doc_buffer1 = (char*)malloc(65536);
    size_t doc_size1 = 65536;
    int r1_final = cactus_index_get(index, &id1, 1, &doc_buffer1, &doc_size1, nullptr, nullptr, nullptr, nullptr);

    bool success = (r2_final == 0) && (r1_final != 0);

    free(doc_buffer2);
    free(doc_buffer1);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Stress Tests
// ============================================================================

bool test_stress_1000_docs() {
    const std::string dir_path = std::string(g_index_path) + "/test_stress_1000";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int batch = 0; batch < 10; ++batch) {
        int ids[100];
        const char* docs[100];
        const char* metas[100];
        std::vector<std::vector<float>> embeddings_data(100);
        const float* embeddings[100];
        std::vector<std::string> doc_strs, meta_strs;

        for (int i = 0; i < 100; ++i) {
            ids[i] = batch * 100 + i;
            doc_strs.push_back("content" + std::to_string(batch * 100 + i));
            meta_strs.push_back("meta" + std::to_string(batch * 100 + i));
            docs[i] = doc_strs[i].c_str();
            metas[i] = meta_strs[i].c_str();
            embeddings_data[i] = random_embedding(1024);
            embeddings[i] = embeddings_data[i].data();
        }

        cactus_index_add(index, ids, docs, metas, embeddings, 100, 1024);
    }

    int get_id = 500;
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index, &get_id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_stress_rapid_add_delete() {
    const std::string dir_path = std::string(g_index_path) + "/test_stress_rapid";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    for (int cycle = 0; cycle < 10; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            int id = cycle * 10 + i;
            const char* doc = "doc";
            const char* meta = "meta";
            auto emb = random_embedding(1024);
            const float* embedding = emb.data();
            cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
        }

        for (int i = 0; i < 5; ++i) {
            int id = cycle * 10 + i;
            cactus_index_delete(index, &id, 1);
        }
    }

    int get_ids[] = {5, 15, 95};
    char* doc_buffers[3];
    for (int i = 0; i < 3; ++i) doc_buffers[i] = (char*)malloc(65536);
    size_t doc_sizes[3] = {65536, 65536, 65536};

    int result = cactus_index_get(index, get_ids, 3, doc_buffers, doc_sizes, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0);

    for (int i = 0; i < 3; ++i) free(doc_buffers[i]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

bool test_edge_add_empty() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_add_empty";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int result = cactus_index_add(index, nullptr, nullptr, nullptr, nullptr, 0, 1024);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_get_nonexistent() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_get_nonexistent";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int add_id = 1;
    const char* doc = "content";
    const char* meta = "meta";
    auto emb = random_embedding(1024);
    const float* embedding = emb.data();
    cactus_index_add(index, &add_id, &doc, &meta, &embedding, 1, 1024);

    int get_id = 999;
    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index, &get_id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result != 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_delete_nonexistent() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_delete_nonexistent";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int delete_id = 999;
    int result = cactus_index_delete(index, &delete_id, 1);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_delete_already_deleted() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_delete_already";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int id = 1;
    const char* doc = "content";
    const char* meta = "meta";
    auto emb = random_embedding(1024);
    const float* embedding = emb.data();
    cactus_index_add(index, &id, &doc, &meta, &embedding, 1, 1024);
    cactus_index_delete(index, &id, 1);

    int result = cactus_index_delete(index, &id, 1);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_query_empty_index() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_query_empty";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    auto query_emb = random_embedding(1024);
    const float* queries[] = {query_emb.data()};
    int* id_buffers[1];
    float* score_buffers[1];
    id_buffers[0] = (int*)malloc(10 * sizeof(int));
    score_buffers[0] = (float*)malloc(10 * sizeof(float));
    size_t id_sizes[1] = {10};
    size_t score_sizes[1] = {10};

    const char* options = "{\"top_k\":10}";
    int result = cactus_index_query(index, queries, 1, 1024, options, id_buffers, id_sizes, score_buffers, score_sizes);

    bool success = (result == 0) && (id_sizes[0] == 0);

    free(id_buffers[0]);
    free(score_buffers[0]);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_zero_embedding() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_zero_embedding";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    std::vector<float> zero_embedding(1024, 0.0f);
    int ids[] = {1};
    const char* docs[] = {"content"};
    const char* metas[] = {"meta"};
    const float* embeddings[] = {zero_embedding.data()};

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_duplicate_id() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_duplicate_id";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs[] = {"content1"};
    const char* metas[] = {"meta1"};
    auto emb1 = random_embedding(1024);
    const float* embeddings[] = {emb1.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);

    auto emb2 = random_embedding(1024);
    const char* docs2[] = {"content2"};
    const float* embeddings2[] = {emb2.data()};

    int result = cactus_index_add(index, ids, docs2, metas, embeddings2, 1, 1024);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_duplicate_id_in_batch() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_duplicate_batch";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1, 2, 1};
    const char* docs[] = {"first", "second", "duplicate"};
    const char* metas[] = {"meta1", "meta2", "meta3"};
    auto emb1 = random_embedding(1024);
    auto emb2 = random_embedding(1024);
    auto emb3 = random_embedding(1024);
    const float* embeddings[] = {emb1.data(), emb2.data(), emb3.data()};

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 3, 1024);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_wrong_dimension() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_wrong_dimension";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    std::vector<float> wrong_dim_embedding(1034, 0.5f);
    int ids[] = {1};
    const char* docs[] = {"doc"};
    const char* metas[] = {"meta"};
    const float* embeddings[] = {wrong_dim_embedding.data()};

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 1, 1034);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_empty_embedding() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_empty_embedding";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs[] = {"doc"};
    const char* metas[] = {"meta"};
    const float* embeddings[] = {nullptr};

    int result = cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
    bool success = (result != 0);

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_nan_embedding() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_nan_embedding";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    std::vector<float> nan_embedding(1024, std::nan(""));
    int ids[] = {1};
    const char* docs[] = {"content"};
    const char* metas[] = {"meta"};
    const float* embeddings[] = {nan_embedding.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
    bool success = true;

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_inf_embedding() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_inf_embedding";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    std::vector<float> inf_embedding(1024, std::numeric_limits<float>::infinity());
    int ids[] = {1};
    const char* docs[] = {"content"};
    const char* metas[] = {"meta"};
    const float* embeddings[] = {inf_embedding.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);
    bool success = true;

    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_negative_doc_id() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_negative_id";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {-1};
    const char* docs[] = {"negative id"};
    const char* metas[] = {"meta"};
    auto emb = random_embedding(1024);
    const float* embeddings[] = {emb.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);

    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

bool test_edge_unicode_content() {
    const std::string dir_path = std::string(g_index_path) + "/test_edge_unicode";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    cactus_index_t index = cactus_index_init(dir_path.c_str(), 1024);
    if (!index) return false;

    int ids[] = {1};
    const char* docs[] = {"Hello 世界 🌍"};
    const char* metas[] = {"méta données"};
    auto emb = random_embedding(1024);
    const float* embeddings[] = {emb.data()};

    cactus_index_add(index, ids, docs, metas, embeddings, 1, 1024);

    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    int result = cactus_index_get(index, ids, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    bool success = (result == 0) && (strcmp(doc_buffer, "Hello 世界 🌍") == 0);

    free(doc_buffer);
    cactus_index_destroy(index);
    cleanup_test_dir(dir_path);
    return success;
}

// ============================================================================
// Benchmark Tests
// ============================================================================

void run_benchmarks(size_t embedding_dim, uint32_t num_docs) {
    std::cout << "\n╔══════════════════════════════════════════════════════════════════════════════════════╗\n"
              << "║ Index Benchmark Suite - " << std::setw(61) << std::left << (std::to_string(num_docs) + " Documents") << "║\n"
              << "╚══════════════════════════════════════════════════════════════════════════════════════╝\n";

    const std::string dir_path = std::string(g_index_path) + "/bench_index";
    cleanup_test_dir(dir_path);
    create_test_dir(dir_path);

    auto start = std::chrono::high_resolution_clock::now();
    cactus_index_t index = cactus_index_init(dir_path.c_str(), embedding_dim);
    auto end = std::chrono::high_resolution_clock::now();

    std::vector<int> all_ids;
    std::vector<const char*> all_docs, all_metas;
    std::vector<std::vector<float>> all_embeddings;
    std::vector<const float*> all_embedding_ptrs;
    std::vector<std::string> doc_strings, meta_strings;

    all_ids.reserve(num_docs);
    doc_strings.reserve(num_docs);
    meta_strings.reserve(num_docs);
    all_embeddings.reserve(num_docs);

    for (uint32_t i = 0; i < num_docs; ++i) {
        all_ids.push_back(static_cast<int>(i));
        doc_strings.push_back("content_" + std::to_string(i));
        meta_strings.push_back("meta_" + std::to_string(i));
        all_embeddings.push_back(random_embedding(embedding_dim));
    }

    for (size_t i = 0; i < num_docs; ++i) {
        all_docs.push_back(doc_strings[i].c_str());
        all_metas.push_back(meta_strings[i].c_str());
        all_embedding_ptrs.push_back(all_embeddings[i].data());
    }

    start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < num_docs; i += 1000) {
        size_t count = std::min(size_t(1000), num_docs - i);
        cactus_index_add(index, all_ids.data() + i, all_docs.data() + i, all_metas.data() + i,
                        all_embedding_ptrs.data() + i, count, embedding_dim);
    }
    end = std::chrono::high_resolution_clock::now();

    cactus_index_destroy(index);

    start = std::chrono::high_resolution_clock::now();
    cactus_index_t index2 = cactus_index_init(dir_path.c_str(), embedding_dim);
    end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    const uint32_t num_adds = std::min(1000u, num_docs / 10);
    std::vector<int> add_ids;
    std::vector<std::string> add_doc_strings, add_meta_strings;
    std::vector<const char*> add_docs, add_metas;
    std::vector<std::vector<float>> add_embeddings;
    std::vector<const float*> add_embedding_ptrs;

    add_ids.reserve(num_adds);
    add_doc_strings.reserve(num_adds);
    add_meta_strings.reserve(num_adds);
    add_embeddings.reserve(num_adds);

    for (uint32_t i = 0; i < num_adds; ++i) {
        add_ids.push_back(static_cast<int>(i + num_docs));
        add_doc_strings.push_back("new_content_" + std::to_string(i));
        add_meta_strings.push_back("new_meta_" + std::to_string(i));
        add_embeddings.push_back(random_embedding(embedding_dim));
    }

    for (size_t i = 0; i < num_adds; ++i) {
        add_docs.push_back(add_doc_strings[i].c_str());
        add_metas.push_back(add_meta_strings[i].c_str());
        add_embedding_ptrs.push_back(add_embeddings[i].data());
    }

    start = std::chrono::high_resolution_clock::now();
    cactus_index_add(index2, add_ids.data(), add_docs.data(), add_metas.data(),
                    add_embedding_ptrs.data(), num_adds, embedding_dim);
    end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    auto query_embedding = random_embedding(embedding_dim);
    const float* query_ptr = query_embedding.data();

    int* result_ids = (int*)malloc(10 * sizeof(int));
    float* result_scores = (float*)malloc(10 * sizeof(float));
    int* ids_ptr[1] = {result_ids};
    float* scores_ptr[1] = {result_scores};
    size_t id_size[1] = {10};
    size_t score_size[1] = {10};

    start = std::chrono::high_resolution_clock::now();
    cactus_index_query(index2, &query_ptr, 1, embedding_dim, "{\"top_k\":10}",
                      ids_ptr, id_size, scores_ptr, score_size);
    end = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    free(result_ids);
    free(result_scores);

    const int num_gets = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> get_doc_ids;
    get_doc_ids.reserve(num_gets);
    for (int i = 0; i < num_gets; ++i) {
        get_doc_ids.push_back(i * (num_docs / num_gets));
    }

    char* doc_buffer = (char*)malloc(65536);
    size_t doc_size = 65536;

    start = std::chrono::high_resolution_clock::now();
    for (const auto& doc_id : get_doc_ids) {
        doc_size = 65536;
        cactus_index_get(index2, &doc_id, 1, &doc_buffer, &doc_size, nullptr, nullptr, nullptr, nullptr);
    }
    end = std::chrono::high_resolution_clock::now();
    auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    free(doc_buffer);

    const int num_deletes = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> delete_doc_ids;
    delete_doc_ids.reserve(num_deletes);
    for (int i = 0; i < num_deletes; ++i) {
        delete_doc_ids.push_back(i);
    }

    start = std::chrono::high_resolution_clock::now();
    cactus_index_delete(index2, delete_doc_ids.data(), num_deletes);
    end = std::chrono::high_resolution_clock::now();
    auto delete_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    start = std::chrono::high_resolution_clock::now();
    cactus_index_compact(index2);
    end = std::chrono::high_resolution_clock::now();
    auto compact_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Benchmark Summary                    ║\n"
              << "╠══════════════════════════════════════════╣\n"
              << "║ Init:    " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << load_duration << "ms ║\n"
              << "║ Add:     " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << add_duration << "ms ║\n"
              << "║ Query:   " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << query_duration << "ms ║\n"
              << "║ Get:     " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << get_duration << "ms ║\n"
              << "║ Delete:  " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << delete_duration << "ms ║\n"
              << "║ Compact: " << std::setw(29) << std::right << std::fixed << std::setprecision(6) << compact_duration << "ms ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cactus_index_destroy(index2);
    cleanup_test_dir(dir_path);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    TestUtils::TestRunner runner("Index FFI Tests");

    // Constructor tests
    runner.run_test("constructor_valid", test_constructor_valid());
    runner.run_test("constructor_missing_index", test_constructor_missing_index());
    runner.run_test("constructor_missing_data", test_constructor_missing_data());
    runner.run_test("constructor_dimension_mismatch", test_constructor_dimension_mismatch());

    // Add document tests
    runner.run_test("add_document", test_add_document());
    runner.run_test("add_multiple_documents", test_add_multiple_documents());
    runner.run_test("add_with_null_metadata", test_add_with_null_metadata());
    runner.run_test("add_after_delete", test_add_after_delete());

    // Get document tests
    runner.run_test("get_document", test_get_document());
    runner.run_test("get_multiple_documents", test_get_multiple_documents());
    runner.run_test("get_only_documents", test_get_only_documents());
    runner.run_test("get_after_compact", test_get_after_compact());

    // Delete tests
    runner.run_test("delete_document", test_delete_document());
    runner.run_test("delete_alternating", test_delete_alternating());
    runner.run_test("delete_then_query", test_delete_then_query());

    // Compact tests
    runner.run_test("compact_reclaim_space", test_compact_reclaim_space());
    runner.run_test("compact_query_after", test_compact_query_after());
    runner.run_test("compact_empty_index", test_compact_empty_index());
    runner.run_test("compact_all_deleted", test_compact_all_deleted());
    runner.run_test("compact_large_gaps", test_compact_large_gaps());

    // Query tests
    runner.run_test("query_similarity", test_query_similarity());
    runner.run_test("query_topk", test_query_topk());
    runner.run_test("query_exact_match", test_query_exact_match());
    runner.run_test("query_score_range", test_query_score_range());
    runner.run_test("query_score_ordering", test_query_score_ordering());
    runner.run_test("query_score_threshold", test_query_score_threshold());
    runner.run_test("query_threshold_default", test_query_threshold_default());
    runner.run_test("query_empty_embeddings", test_query_empty_embeddings());
    runner.run_test("query_batch", test_query_batch());

    // Persistence tests
    runner.run_test("persist_after_add", test_persist_after_add());
    runner.run_test("persist_after_delete", test_persist_after_delete());
    runner.run_test("persist_after_compact", test_persist_after_compact());
    runner.run_test("persist_reload_sequence", test_persist_reload_sequence());

    // Stress tests
    runner.run_test("stress_1000_docs", test_stress_1000_docs());
    runner.run_test("stress_rapid_add_delete", test_stress_rapid_add_delete());

    // Edge case tests
    runner.run_test("edge_add_empty", test_edge_add_empty());
    runner.run_test("edge_get_nonexistent", test_edge_get_nonexistent());
    runner.run_test("edge_delete_nonexistent", test_edge_delete_nonexistent());
    runner.run_test("edge_delete_already_deleted", test_edge_delete_already_deleted());
    runner.run_test("edge_query_empty_index", test_edge_query_empty_index());
    runner.run_test("edge_zero_embedding", test_edge_zero_embedding());
    runner.run_test("edge_duplicate_id", test_edge_duplicate_id());
    runner.run_test("edge_duplicate_id_in_batch", test_edge_duplicate_id_in_batch());
    runner.run_test("edge_wrong_dimension", test_edge_wrong_dimension());
    runner.run_test("edge_empty_embedding", test_edge_empty_embedding());
    runner.run_test("edge_nan_embedding", test_edge_nan_embedding());
    runner.run_test("edge_inf_embedding", test_edge_inf_embedding());
    runner.run_test("edge_negative_doc_id", test_edge_negative_doc_id());
    runner.run_test("edge_unicode_content", test_edge_unicode_content());

    runner.print_summary();

    // Benchmarks
    const size_t embedding_dim = 1024;
    const uint32_t num_documents = 100000;
    run_benchmarks(embedding_dim, num_documents);

    return runner.all_passed() ? 0 : 1;
}

#include "../cactus/index/index.h"
#include "test_utils.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>

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

void cleanup_test_files(const std::string& index_path, const std::string& data_path) {
    unlink(index_path.c_str());
    unlink(data_path.c_str());
}

bool expect_exception(std::function<void()> func) {
    try {
        func();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

// ============================================================================
// Constructor Tests
// ============================================================================

bool test_constructor_valid() {
    const std::string index_path = std::string(g_index_path) + "/test_constructor_valid.idx";
    const std::string data_path = std::string(g_index_path) + "/test_constructor_valid.dat";
    cleanup_test_files(index_path, data_path);

    bool success = false;
    try {
        cactus::index::Index index(index_path, data_path, 1024);
        success = true;
    } catch (...) {
        success = false;
    }
    cleanup_test_files(index_path, data_path);
    return success;
}

bool test_constructor_creates_new_files() {
    const std::string index_path = std::string(g_index_path) + "/test_creates_new.idx";
    const std::string data_path = std::string(g_index_path) + "/test_creates_new.dat";
    cleanup_test_files(index_path, data_path);

    bool success = false;
    try {
        cactus::index::Index index(index_path, data_path, 1024);
        success = (access(index_path.c_str(), F_OK) == 0) &&
                  (access(data_path.c_str(), F_OK) == 0);
    } catch (...) {
        success = false;
    }

    cleanup_test_files(index_path, data_path);
    return success;
}

bool test_constructor_missing_index() {
    const std::string index_path = std::string(g_index_path) + "/test_missing.idx";
    const std::string data_path = std::string(g_index_path) + "/test_missing.dat";
    cleanup_test_files(index_path, data_path);

    std::ofstream data_file(data_path, std::ios::binary);
    data_file.close();

    bool caught = expect_exception([&]() {
        cactus::index::Index index(index_path, data_path, 1024);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_constructor_missing_data() {
    const std::string index_path = std::string(g_index_path) + "/test_missing_data.idx";
    const std::string data_path = std::string(g_index_path) + "/test_missing_data.dat";
    cleanup_test_files(index_path, data_path);

    std::ofstream index_file(index_path, std::ios::binary);
    index_file.close();

    bool caught = expect_exception([&]() {
        cactus::index::Index index(index_path, data_path, 1024);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_constructor_wrong_magic() {
    const std::string index_path = std::string(g_index_path) + "/test_wrong_magic.idx";
    const std::string data_path = std::string(g_index_path) + "/test_wrong_magic.dat";
    cleanup_test_files(index_path, data_path);

    std::ofstream index_file(index_path, std::ios::binary);
    uint32_t wrong_magic = 0xDEADBEEF;
    index_file.write(reinterpret_cast<const char*>(&wrong_magic), sizeof(uint32_t));
    index_file.close();

    std::ofstream data_file(data_path, std::ios::binary);
    data_file.close();

    bool caught = expect_exception([&]() {
        cactus::index::Index index(index_path, data_path, 1024);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_constructor_dimension_mismatch() {
    const std::string index_path = std::string(g_index_path) + "/test_dim_mismatch.idx";
    const std::string data_path = std::string(g_index_path) + "/test_dim_mismatch.dat";
    cleanup_test_files(index_path, data_path);

    {
        cactus::index::Index index(index_path, data_path, 1024);
    }

    bool caught = expect_exception([&]() {
        cactus::index::Index index(index_path, data_path, 256);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

// ============================================================================
// Add Document Tests
// ============================================================================

bool test_add_document() {
    const std::string index_path = std::string(g_index_path) + "/test_add_document.idx";
    const std::string data_path = std::string(g_index_path) + "/test_add_document.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content1", "metadata1"}
    };

    index.add_documents(docs);
    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].id == 1;
}

bool test_add_multiple_documents() {
    const std::string index_path = std::string(g_index_path) + "/test_add_multiple.idx";
    const std::string data_path = std::string(g_index_path) + "/test_add_multiple.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({
            i,
            random_embedding(1024),
            "content" + std::to_string(i),
            "metadata" + std::to_string(i)
        });
    }

    index.add_documents(docs);
    auto docs_result = index.get_documents({0, 5, 9});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 3 &&
           docs_result[0].id == 0 &&
           docs_result[1].id == 5 &&
           docs_result[2].id == 9;
}

bool test_add_after_delete() {
    const std::string index_path = std::string(g_index_path) + "/test_add_after_delete.idx";
    const std::string data_path = std::string(g_index_path) + "/test_add_after_delete.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    index.add_documents({{1, random_embedding(1024), "first", "meta1"}});
    index.delete_documents({1});
    index.add_documents({{1, random_embedding(1024), "second", "meta2"}});

    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].content == "second";
}

// ============================================================================
// Get Document Tests
// ============================================================================

bool test_get_document() {
    const std::string index_path = std::string(g_index_path) + "/test_get_document.idx";
    const std::string data_path = std::string(g_index_path) + "/test_get_document.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    auto embedding = random_embedding(1024);
    std::vector<cactus::index::Document> docs = {
        {1, embedding, "test content", "test metadata"}
    };
    index.add_documents(docs);

    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 &&
           docs_result[0].id == 1 &&
           docs_result[0].content == "test content" &&
           docs_result[0].metadata == "test metadata" &&
           docs_result[0].embedding.size() == 1024;
}

bool test_get_multiple_documents() {
    const std::string index_path = std::string(g_index_path) + "/test_get_multiple.idx";
    const std::string data_path = std::string(g_index_path) + "/test_get_multiple.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({
            i,
            random_embedding(1024),
            "content" + std::to_string(i),
            "meta" + std::to_string(i)
        });
    }
    index.add_documents(docs);

    auto docs_result = index.get_documents({2, 5, 8});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 3 &&
           docs_result[0].id == 2 &&
           docs_result[0].content == "content2" &&
           docs_result[1].id == 5 &&
           docs_result[1].content == "content5" &&
           docs_result[2].id == 8 &&
           docs_result[2].content == "content8";
}

bool test_get_after_compact() {
    const std::string index_path = std::string(g_index_path) + "/test_get_after_compact.idx";
    const std::string data_path = std::string(g_index_path) + "/test_get_after_compact.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({
            i,
            random_embedding(1024),
            "content" + std::to_string(i),
            "meta" + std::to_string(i)
        });
    }
    index.add_documents(docs);

    index.delete_documents({1, 3, 5, 7, 9});
    index.compact();

    auto docs_result = index.get_documents({0, 2, 4, 6, 8});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 5 &&
           docs_result[0].content == "content0" &&
           docs_result[1].content == "content2" &&
           docs_result[2].content == "content4" &&
           docs_result[3].content == "content6" &&
           docs_result[4].content == "content8";
}

// ============================================================================
// Delete Tests
// ============================================================================

bool test_delete_document() {
    const std::string index_path = std::string(g_index_path) + "/test_delete_document.idx";
    const std::string data_path = std::string(g_index_path) + "/test_delete_document.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content1", "metadata1"},
        {2, random_embedding(1024), "content2", "metadata2"}
    };
    index.add_documents(docs);
    index.delete_documents({1});

    bool caught = expect_exception([&]() {
        index.get_documents({1});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_delete_alternating() {
    const std::string index_path = std::string(g_index_path) + "/test_delete_alternating.idx";
    const std::string data_path = std::string(g_index_path) + "/test_delete_alternating.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 20; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }

    std::vector<int> to_delete;
    for (int i = 0; i < 20; i += 2) to_delete.push_back(i);
    index.delete_documents(to_delete);

    bool all_throw = true;
    for (int id : {0, 2, 4, 6}) {
        if (!expect_exception([&]() { index.get_documents({id}); })) {
            all_throw = false;
        }
    }

    cleanup_test_files(index_path, data_path);
    return all_throw;
}

bool test_delete_then_query() {
    const std::string index_path = std::string(g_index_path) + "/test_delete_then_query.idx";
    const std::string data_path = std::string(g_index_path) + "/test_delete_then_query.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 10; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }
    index.delete_documents({0, 1, 2});

    auto res = index.query({random_embedding(1024)}, {10, 0.0f});

    bool pass = false;
    if (!res.empty() && res[0].size() <= 7) {
        pass = true;
        for (const auto& result : res[0]) {
            if (result.doc_id == 0 || result.doc_id == 1 || result.doc_id == 2) {
                pass = false;
                break;
            }
        }
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

// ============================================================================
// Compact Tests
// ============================================================================

bool test_compact_reclaim_space() {
    const std::string index_path = std::string(g_index_path) + "/test_compact_reclaim.idx";
    const std::string data_path = std::string(g_index_path) + "/test_compact_reclaim.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({
            i,
            random_embedding(1024),
            "content" + std::to_string(i),
            "metadata" + std::to_string(i)
        });
    }
    index.add_documents(docs);

    struct stat st_before;
    stat(index_path.c_str(), &st_before);
    size_t index_size_before = st_before.st_size;

    index.delete_documents({0, 2, 4, 6, 8});
    index.compact();

    struct stat st_after;
    stat(index_path.c_str(), &st_after);
    size_t index_size_after = st_after.st_size;

    auto docs_result = index.get_documents({1, 3, 5, 7, 9});

    bool size_reduced = index_size_after < index_size_before;
    bool correct_docs = docs_result.size() == 5 &&
                       docs_result[0].id == 1 &&
                       docs_result[1].id == 3 &&
                       docs_result[2].id == 5 &&
                       docs_result[3].id == 7 &&
                       docs_result[4].id == 9;

    bool deleted_docs_gone = expect_exception([&]() {
        index.get_documents({0});
    }) && expect_exception([&]() {
        index.get_documents({2});
    });

    cleanup_test_files(index_path, data_path);
    return size_reduced && correct_docs && deleted_docs_gone;
}

bool test_compact_query_after() {
    const std::string index_path = std::string(g_index_path) + "/test_compact_query.idx";
    const std::string data_path = std::string(g_index_path) + "/test_compact_query.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    auto embedding1 = random_embedding(1024);
    auto embedding2 = random_embedding(1024);
    auto embedding3 = random_embedding(1024);

    std::vector<cactus::index::Document> docs = {
        {1, embedding1, "content1", "metadata1"},
        {2, embedding2, "content2", "metadata2"},
        {3, embedding3, "content3", "metadata3"}
    };
    index.add_documents(docs);

    index.delete_documents({2});
    index.compact();

    cactus::index::QueryOptions options;
    options.top_k = 2;

    auto res = index.query({embedding1}, options);
    bool query_works = res.size() == 1 && res[0].size() >= 1 && res[0][0].doc_id == 1;

    auto docs_result = index.get_documents({1});
    bool content_preserved = docs_result[0].content == "content1" &&
                            docs_result[0].metadata == "metadata1";

    cleanup_test_files(index_path, data_path);
    return query_works && content_preserved;
}

bool test_compact_empty_index() {
    const std::string index_path = std::string(g_index_path) + "/test_compact_empty.idx";
    const std::string data_path = std::string(g_index_path) + "/test_compact_empty.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    index.compact();

    cleanup_test_files(index_path, data_path);
    return true;
}

bool test_compact_all_deleted() {
    const std::string index_path = std::string(g_index_path) + "/test_compact_all_deleted.idx";
    const std::string data_path = std::string(g_index_path) + "/test_compact_all_deleted.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 5; ++i) {
        docs.push_back({i, random_embedding(1024), "content", "meta"});
    }
    index.add_documents(docs);
    index.delete_documents({0, 1, 2, 3, 4});

    index.compact();

    struct stat st;
    stat(index_path.c_str(), &st);
    size_t index_size = st.st_size;

    size_t header_size = 16;

    cleanup_test_files(index_path, data_path);
    return index_size == header_size;
}

bool test_compact_large_gaps() {
    const std::string index_path = std::string(g_index_path) + "/test_compact_gaps.idx";
    const std::string data_path = std::string(g_index_path) + "/test_compact_gaps.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 100; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }
    for (int i = 10; i < 90; ++i) {
        index.delete_documents({i});
    }

    index.compact();

    auto docs_result = index.get_documents({0, 5, 90, 95});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 4;
}

// ============================================================================
// Query Tests
// ============================================================================

bool test_query_similarity() {
    const std::string index_path = std::string(g_index_path) + "/test_query_similarity.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_similarity.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    auto embedding1 = random_embedding(1024);
    auto embedding2 = random_embedding(1024);

    std::vector<cactus::index::Document> docs = {
        {1, embedding1, "content1", "metadata1"},
        {2, embedding2, "content2", "metadata2"}
    };
    index.add_documents(docs);

    cactus::index::QueryOptions options;
    options.top_k = 1;

    auto res = index.query({embedding1}, options);

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].size() == 1 && res[0][0].doc_id == 1;
}

bool test_query_topk() {
    const std::string index_path = std::string(g_index_path) + "/test_query_topk.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_topk.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({i, random_embedding(1024), "content", "meta"});
    }
    index.add_documents(docs);

    cactus::index::QueryOptions options;
    options.top_k = 5;

    auto res = index.query({random_embedding(1024)}, options);

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].size() <= 5 && res[0].size() > 0;
}

bool test_query_exact_match() {
    const std::string index_path = std::string(g_index_path) + "/test_query_exact.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_exact.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    auto embedding = random_embedding(1024);
    index.add_documents({{1, embedding, "exact", "meta"}});
    for (int i = 2; i < 10; ++i) {
        index.add_documents({{i, random_embedding(1024), "other", "meta"}});
    }

    auto res = index.query({embedding}, {1, 0.0f});

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].size() == 1 && res[0][0].doc_id == 1;
}

bool test_query_score_range() {
    const std::string index_path = std::string(g_index_path) + "/test_query_score_range.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_score_range.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 10; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }

    auto res = index.query({random_embedding(1024)}, {10, 0.0f});

    bool pass = true;
    if (res.empty() || res[0].empty()) {
        pass = false;
    } else {
        for (const auto& result : res[0]) {
            if (result.score < -1.0f || result.score > 1.0f) {
                pass = false;
                break;
            }
        }
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

bool test_query_score_ordering() {
    const std::string index_path = std::string(g_index_path) + "/test_query_ordering.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_ordering.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 1; i <= 20; i++) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }

    auto res = index.query({random_embedding(1024)}, {.top_k = 10});

    bool pass = true;
    for (size_t i = 1; i < res[0].size(); i++) {
        if (res[0][i-1].score < res[0][i].score) {
            pass = false;
            break;
        }
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

bool test_query_score_threshold() {
    const std::string index_path = std::string(g_index_path) + "/test_query_threshold.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_threshold.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    auto embedding = random_embedding(1024);
    index.add_documents({{1, embedding, "exact", "meta"}});
    for (int i = 2; i < 10; ++i) {
        index.add_documents({{i, random_embedding(1024), "other", "meta"}});
    }

    auto res = index.query({embedding}, {.top_k = 10, .score_threshold = 0.95f});

    bool pass = false;
    if (!res.empty() && !res[0].empty()) {
        pass = true;
        for (const auto& result : res[0]) {
            if (result.score < 0.95f) {
                pass = false;
                break;
            }
        }
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

bool test_query_threshold_none_match() {
    const std::string index_path = std::string(g_index_path) + "/test_query_threshold_none.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_threshold_none.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 5; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }

    auto res = index.query({random_embedding(1024)}, {.top_k = 10, .score_threshold = 1.5f});

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].empty();
}

bool test_query_threshold_default() {
    const std::string index_path = std::string(g_index_path) + "/test_query_threshold_default.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_threshold_default.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int i = 0; i < 5; ++i) {
        index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
    }

    auto res1 = index.query({random_embedding(1024)}, {.top_k = 5});
    auto res2 = index.query({random_embedding(1024)}, {.top_k = 5, .score_threshold = -1.0f});

    cleanup_test_files(index_path, data_path);
    return res1.size() == 1 && res2.size() == 1 &&
           res1[0].size() == res2[0].size();
}

bool test_query_empty_embeddings() {
    const std::string index_path = std::string(g_index_path) + "/test_query_empty_embeddings.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_empty_embeddings.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    index.add_documents({{1, random_embedding(1024), "doc", "meta"}});

    std::vector<std::vector<float>> empty_queries;
    auto res = index.query(empty_queries, {.top_k = 10});

    cleanup_test_files(index_path, data_path);
    return res.empty();
}

bool test_query_zero_topk() {
    const std::string index_path = std::string(g_index_path) + "/test_query_zero_topk.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_zero_topk.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    index.add_documents({{1, random_embedding(1024), "doc", "meta"}});
    auto res = index.query({random_embedding(1024)}, {0, 0.0f});

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].empty();
}

bool test_query_batch() {
    const std::string index_path = std::string(g_index_path) + "/test_query_batch.idx";
    const std::string data_path = std::string(g_index_path) + "/test_query_batch.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 10; ++i) {
        docs.push_back({i, random_embedding(1024), "content", "meta"});
    }
    index.add_documents(docs);

    std::vector<std::vector<float>> queries;
    for (int i = 0; i < 5; ++i) {
        queries.push_back(random_embedding(1024));
    }

    cactus::index::QueryOptions options;
    options.top_k = 3;

    auto res = index.query(queries, options);

    cleanup_test_files(index_path, data_path);
    return res.size() == 5 && res[0].size() <= 3;
}

// ============================================================================
// Persistence Tests
// ============================================================================

bool test_persist_after_add() {
    const std::string index_path = std::string(g_index_path) + "/test_persist_add.idx";
    const std::string data_path = std::string(g_index_path) + "/test_persist_add.dat";
    cleanup_test_files(index_path, data_path);

    {
        cactus::index::Index index(index_path, data_path, 1024);
        index.add_documents({{1, random_embedding(1024), "persisted", "meta"}});
    }

    cactus::index::Index index2(index_path, data_path, 1024);
    auto docs_result = index2.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].content == "persisted";
}

bool test_persist_after_delete() {
    const std::string index_path = std::string(g_index_path) + "/test_persist_delete.idx";
    const std::string data_path = std::string(g_index_path) + "/test_persist_delete.dat";
    cleanup_test_files(index_path, data_path);

    {
        cactus::index::Index index(index_path, data_path, 1024);
        index.add_documents({{1, random_embedding(1024), "doc", "meta"}});
        index.delete_documents({1});
    }

    cactus::index::Index index2(index_path, data_path, 1024);
    bool caught = expect_exception([&]() {
        index2.get_documents({1});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_persist_after_compact() {
    const std::string index_path = std::string(g_index_path) + "/test_persist_compact.idx";
    const std::string data_path = std::string(g_index_path) + "/test_persist_compact.dat";
    cleanup_test_files(index_path, data_path);

    {
        cactus::index::Index index(index_path, data_path, 1024);
        for (int i = 0; i < 10; ++i) {
            index.add_documents({{i, random_embedding(1024), "doc", "meta"}});
        }
        index.delete_documents({0, 1, 2});
        index.compact();
    }

    cactus::index::Index index2(index_path, data_path, 1024);
    auto docs_result = index2.get_documents({3});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].id == 3;
}

bool test_persist_reload_sequence() {
    const std::string index_path = std::string(g_index_path) + "/test_persist_sequence.idx";
    const std::string data_path = std::string(g_index_path) + "/test_persist_sequence.dat";
    cleanup_test_files(index_path, data_path);

    {
        cactus::index::Index index(index_path, data_path, 1024);
        index.add_documents({{1, random_embedding(1024), "doc1", "meta"}});
    }

    {
        cactus::index::Index index(index_path, data_path, 1024);
        auto r1 = index.get_documents({1});
        if (r1.size() != 1) return false;
        index.add_documents({{2, random_embedding(1024), "doc2", "meta"}});
    }

    {
        cactus::index::Index index(index_path, data_path, 1024);
        auto r2 = index.get_documents({2});
        if (r2.size() != 1) return false;
        index.delete_documents({1});
    }

    {
        cactus::index::Index index(index_path, data_path, 1024);
        index.compact();
    }

    bool pass = false;
    {
        cactus::index::Index index(index_path, data_path, 1024);
        auto r2 = index.get_documents({2});
        pass = r2.size() == 1 && expect_exception([&]() { index.get_documents({1}); });
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

// ============================================================================
// Stress Tests
// ============================================================================

bool test_stress_1000_docs() {
    const std::string index_path = std::string(g_index_path) + "/test_stress_1000.idx";
    const std::string data_path = std::string(g_index_path) + "/test_stress_1000.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    for (int i = 0; i < 1000; ++i) {
        docs.push_back({
            i,
            random_embedding(1024),
            "content" + std::to_string(i),
            "meta" + std::to_string(i)
        });
    }

    index.add_documents(docs);
    auto docs_result = index.get_documents({500});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].id == 500;
}

bool test_stress_rapid_add_delete() {
    const std::string index_path = std::string(g_index_path) + "/test_stress_rapid.idx";
    const std::string data_path = std::string(g_index_path) + "/test_stress_rapid.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    for (int cycle = 0; cycle < 10; ++cycle) {
        for (int i = 0; i < 10; ++i) {
            index.add_documents({{cycle * 10 + i, random_embedding(1024), "doc", "meta"}});
        }
        for (int i = 0; i < 5; ++i) {
            index.delete_documents({cycle * 10 + i});
        }
    }

    auto docs_result = index.get_documents({5, 15, 95});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 3;
}

// ============================================================================
// Edge Case Tests
// ============================================================================

bool test_edge_add_empty() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_add_empty.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_add_empty.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs;
    bool caught = expect_exception([&]() {
        index.add_documents(docs);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_content_over_limit() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_content_limit.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_content_limit.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::string over_limit(65536, 'X');
    bool caught = expect_exception([&]() {
        index.add_documents({{1, random_embedding(1024), over_limit, "meta"}});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_get_nonexistent() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_get_nonexistent.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_get_nonexistent.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content", "meta"}
    };
    index.add_documents(docs);

    bool caught = expect_exception([&]() {
        index.get_documents({999});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_get_empty() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_get_empty.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_get_empty.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content", "meta"}
    };
    index.add_documents(docs);

    bool caught = expect_exception([&]() {
        index.get_documents({});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_delete_nonexistent() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_delete_nonexistent.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_delete_nonexistent.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    bool caught = expect_exception([&]() {
        index.delete_documents({999});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_delete_already_deleted() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_delete_already.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_delete_already.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content", "meta"}
    };
    index.add_documents(docs);
    index.delete_documents({1});

    bool caught = expect_exception([&]() {
        index.delete_documents({1});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_delete_empty() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_delete_empty.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_delete_empty.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    index.add_documents({{1, random_embedding(1024), "doc", "meta"}});

    std::vector<int> empty_ids;
    bool caught = expect_exception([&]() {
        index.delete_documents(empty_ids);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_query_empty_index() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_query_empty.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_query_empty.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    cactus::index::QueryOptions options;
    options.top_k = 10;

    auto res = index.query({random_embedding(1024)}, options);

    cleanup_test_files(index_path, data_path);
    return res.size() == 1 && res[0].empty();
}

bool test_edge_zero_embedding() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_zero_embedding.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_zero_embedding.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<float> zero_embedding(1024, 0.0f);
    std::vector<cactus::index::Document> docs = {
        {1, zero_embedding, "content", "meta"}
    };

    bool caught = expect_exception([&]() {
        index.add_documents(docs);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_duplicate_id() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_duplicate_id.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_duplicate_id.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs1 = {
        {1, random_embedding(1024), "content1", "meta1"}
    };
    index.add_documents(docs1);

    std::vector<cactus::index::Document> docs2 = {
        {1, random_embedding(1024), "content2", "meta2"}
    };

    bool caught = expect_exception([&]() {
        index.add_documents(docs2);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_duplicate_id_in_batch() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_duplicate_batch.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_duplicate_batch.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "first", "meta1"},
        {2, random_embedding(1024), "second", "meta2"},
        {1, random_embedding(1024), "duplicate", "meta3"}
    };

    bool caught = expect_exception([&]() {
        index.add_documents(docs);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_max_content_size() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_max_content.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_max_content.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::string max_content(65535, 'x');
    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), max_content, "meta"}
    };

    index.add_documents(docs);
    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].content.size() == 65535;
}

bool test_edge_metadata_too_large() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_metadata_large.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_metadata_large.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::string large_metadata(1024 * 1024, 'x');
    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content", large_metadata}
    };

    bool caught = expect_exception([&]() {
        index.add_documents(docs);
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_metadata_over_limit() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_metadata_limit.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_metadata_limit.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::string over_limit(65536, 'X');
    bool caught = expect_exception([&]() {
        index.add_documents({{1, random_embedding(1024), "content", over_limit}});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_max_metadata_size() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_max_metadata.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_max_metadata.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::string max_metadata(65535, 'x');
    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "content", max_metadata}
    };

    index.add_documents(docs);
    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].metadata.size() == 65535;
}

bool test_edge_wrong_dimension() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_wrong_dimension.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_wrong_dimension.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<float> wrong_dim_embedding(1034, 0.5f);
    bool caught = expect_exception([&]() {
        index.add_documents({{1, wrong_dim_embedding, "doc", "meta"}});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_empty_embedding() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_empty_embedding.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_empty_embedding.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<float> empty_embedding;
    bool caught = expect_exception([&]() {
        index.add_documents({{1, empty_embedding, "doc", "meta"}});
    });

    cleanup_test_files(index_path, data_path);
    return caught;
}

bool test_edge_nan_embedding() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_nan_embedding.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_nan_embedding.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<float> nan_embedding(1024, std::nan(""));
    std::vector<cactus::index::Document> docs = {
        {1, nan_embedding, "content", "meta"}
    };

    bool pass = false;
    try {
        index.add_documents(docs);
        pass = true;
    } catch (const std::runtime_error&) {
        pass = true;
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

bool test_edge_inf_embedding() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_inf_embedding.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_inf_embedding.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<float> inf_embedding(1024, std::numeric_limits<float>::infinity());
    std::vector<cactus::index::Document> docs = {
        {1, inf_embedding, "content", "meta"}
    };

    bool pass = false;
    try {
        index.add_documents(docs);
        pass = true;
    } catch (const std::runtime_error&) {
        pass = true;
    }

    cleanup_test_files(index_path, data_path);
    return pass;
}

bool test_edge_negative_doc_id() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_negative_id.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_negative_id.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {-1, random_embedding(1024), "negative id", "meta"}
    };

    index.add_documents(docs);
    auto docs_result = index.get_documents({-1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].id == -1;
}

bool test_edge_empty_content_and_metadata() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_empty_content.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_empty_content.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    std::vector<cactus::index::Document> docs = {
        {1, random_embedding(1024), "", ""}
    };

    index.add_documents(docs);
    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 &&
           docs_result[0].content.empty() &&
           docs_result[0].metadata.empty();
}

bool test_edge_unicode_content() {
    const std::string index_path = std::string(g_index_path) + "/test_edge_unicode.idx";
    const std::string data_path = std::string(g_index_path) + "/test_edge_unicode.dat";
    cleanup_test_files(index_path, data_path);

    cactus::index::Index index(index_path, data_path, 1024);

    cactus::index::Document doc{1, random_embedding(1024), "Hello 世界 🌍", "méta données"};
    index.add_documents({doc});
    auto docs_result = index.get_documents({1});

    cleanup_test_files(index_path, data_path);
    return docs_result.size() == 1 && docs_result[0].content == "Hello 世界 🌍";
}

// ============================================================================
// Benchmark Tests
// ============================================================================

void run_benchmarks(size_t embedding_dim, uint32_t num_docs) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Index Benchmark Suite                ║\n"
              << "║     Documents: " << std::setw(26) << std::left << num_docs << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    const std::string index_path = std::string(g_index_path) + "/bench_index.idx";
    const std::string data_path = std::string(g_index_path) + "/bench_data.dat";
    cleanup_test_files(index_path, data_path);

    std::cout << "\n[INITIALIZATION: Creating and populating index]\n";

    auto start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index(index_path, data_path, embedding_dim);
    auto end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "├─ Empty index creation: " << std::fixed << std::setprecision(6) << init_duration << "ms\n";

    std::vector<cactus::index::Document> docs;
    docs.reserve(num_docs);
    for (uint32_t i = 0; i < num_docs; ++i) {
        docs.push_back({
            static_cast<int>(i),
            random_embedding(embedding_dim),
            "content_" + std::to_string(i),
            "meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(docs);
    end = std::chrono::high_resolution_clock::now();
    auto populate_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    std::cout << "├─ Populate " << num_docs << " docs: " << std::fixed << std::setprecision(6) << populate_duration << "ms\n"
              << "└─ Total initialization: " << std::fixed << std::setprecision(6) << (init_duration + populate_duration) << "ms\n";

    std::cout << "\n[BENCHMARK: Init - Load existing index]\n";
    start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index2(index_path, data_path, embedding_dim);
    end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "└─ Time: " << std::fixed << std::setprecision(6) << load_duration << "ms\n";

    std::cout << "\n[BENCHMARK: Add - Incremental document addition]\n";
    const uint32_t num_adds = std::min(1000u, num_docs / 10);
    std::vector<cactus::index::Document> new_docs;
    new_docs.reserve(num_adds);
    for (uint32_t i = 0; i < num_adds; ++i) {
        new_docs.push_back({
            static_cast<int>(i + num_docs),
            random_embedding(embedding_dim),
            "new_content_" + std::to_string(i),
            "new_meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(new_docs);
    end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "└─ Time: " << std::fixed << std::setprecision(6) << add_duration << "ms\n";

    std::cout << "\n[BENCHMARK: Query - Similarity search]\n";
    cactus::index::QueryOptions options;
    options.top_k = 10;
    auto query_embedding = random_embedding(embedding_dim);

    start = std::chrono::high_resolution_clock::now();
    index.query({query_embedding}, options);
    end = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "└─ Time: " << std::fixed << std::setprecision(6) << query_duration << "ms\n";

    std::cout << "\n[BENCHMARK: Get - Retrieve documents by ID]\n";
    const int num_gets = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> get_doc_ids;
    get_doc_ids.reserve(num_gets);
    for (int i = 0; i < num_gets; ++i) {
        get_doc_ids.push_back(i * (num_docs / num_gets));
    }

    start = std::chrono::high_resolution_clock::now();
    for (const auto& doc_id : get_doc_ids) {
        index.get_documents({doc_id});
    }
    end = std::chrono::high_resolution_clock::now();
    auto get_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "└─ Time: " << std::fixed << std::setprecision(6) << get_duration << "ms\n";

    std::cout << "\n[BENCHMARK: Delete - Remove documents]\n";
    const int num_deletes = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<int> delete_doc_ids;
    delete_doc_ids.reserve(num_deletes);
    for (int i = 0; i < num_deletes; ++i) {
        delete_doc_ids.push_back(i);
    }

    start = std::chrono::high_resolution_clock::now();
    index.delete_documents(delete_doc_ids);
    end = std::chrono::high_resolution_clock::now();
    auto delete_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    std::cout << "└─ Time: " << std::fixed << std::setprecision(6) << delete_duration << "ms\n";

    std::cout << "\n[BENCHMARK: Compact - Reclaim deleted space]\n";
    struct stat st_before;
    stat(index_path.c_str(), &st_before);
    size_t index_size_before = st_before.st_size;

    start = std::chrono::high_resolution_clock::now();
    index.compact();
    end = std::chrono::high_resolution_clock::now();
    auto compact_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;

    struct stat st_after;
    stat(index_path.c_str(), &st_after);
    size_t index_size_after = st_after.st_size;
    size_t space_reclaimed = index_size_before - index_size_after;

    std::cout << "├─ Time: " << std::fixed << std::setprecision(6) << compact_duration << "ms\n"
              << "├─ Size before: " << index_size_before << " bytes\n"
              << "├─ Size after:  " << index_size_after << " bytes\n"
              << "└─ Space reclaimed: " << space_reclaimed << " bytes\n";

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

    cleanup_test_files(index_path, data_path);
}

// ============================================================================
// Main
// ============================================================================

int main() {
    const size_t embedding_dim = 1024;
    const uint32_t num_documents = 100000;

    TestUtils::TestRunner runner("Index Tests");

    runner.run_test("constructor_valid", test_constructor_valid());
    runner.run_test("constructor_creates_new_files", test_constructor_creates_new_files());
    runner.run_test("constructor_missing_index", test_constructor_missing_index());
    runner.run_test("constructor_missing_data", test_constructor_missing_data());
    runner.run_test("constructor_wrong_magic", test_constructor_wrong_magic());
    runner.run_test("constructor_dimension_mismatch", test_constructor_dimension_mismatch());
    runner.run_test("add_document", test_add_document());
    runner.run_test("add_multiple_documents", test_add_multiple_documents());
    runner.run_test("add_after_delete", test_add_after_delete());
    runner.run_test("get_document", test_get_document());
    runner.run_test("get_multiple_documents", test_get_multiple_documents());
    runner.run_test("get_after_compact", test_get_after_compact());
    runner.run_test("delete_document", test_delete_document());
    runner.run_test("delete_alternating", test_delete_alternating());
    runner.run_test("delete_then_query", test_delete_then_query());
    runner.run_test("compact_reclaim_space", test_compact_reclaim_space());
    runner.run_test("compact_query_after", test_compact_query_after());
    runner.run_test("compact_empty_index", test_compact_empty_index());
    runner.run_test("compact_all_deleted", test_compact_all_deleted());
    runner.run_test("compact_large_gaps", test_compact_large_gaps());
    runner.run_test("query_similarity", test_query_similarity());
    runner.run_test("query_topk", test_query_topk());
    runner.run_test("query_exact_match", test_query_exact_match());
    runner.run_test("query_score_range", test_query_score_range());
    runner.run_test("query_score_ordering", test_query_score_ordering());
    runner.run_test("query_score_threshold", test_query_score_threshold());
    runner.run_test("query_threshold_none_match", test_query_threshold_none_match());
    runner.run_test("query_threshold_default", test_query_threshold_default());
    runner.run_test("query_empty_embeddings", test_query_empty_embeddings());
    runner.run_test("query_zero_topk", test_query_zero_topk());
    runner.run_test("query_batch", test_query_batch());
    runner.run_test("persist_after_add", test_persist_after_add());
    runner.run_test("persist_after_delete", test_persist_after_delete());
    runner.run_test("persist_after_compact", test_persist_after_compact());
    runner.run_test("persist_reload_sequence", test_persist_reload_sequence());
    runner.run_test("stress_1000_docs", test_stress_1000_docs());
    runner.run_test("stress_rapid_add_delete", test_stress_rapid_add_delete());
    runner.run_test("edge_add_empty", test_edge_add_empty());
    runner.run_test("edge_content_over_limit", test_edge_content_over_limit());
    runner.run_test("edge_get_nonexistent", test_edge_get_nonexistent());
    runner.run_test("edge_get_empty", test_edge_get_empty());
    runner.run_test("edge_delete_nonexistent", test_edge_delete_nonexistent());
    runner.run_test("edge_delete_already_deleted", test_edge_delete_already_deleted());
    runner.run_test("edge_delete_empty", test_edge_delete_empty());
    runner.run_test("edge_query_empty_index", test_edge_query_empty_index());
    runner.run_test("edge_zero_embedding", test_edge_zero_embedding());
    runner.run_test("edge_duplicate_id", test_edge_duplicate_id());
    runner.run_test("edge_duplicate_id_in_batch", test_edge_duplicate_id_in_batch());
    runner.run_test("edge_max_content_size", test_edge_max_content_size());
    runner.run_test("edge_metadata_too_large", test_edge_metadata_too_large());
    runner.run_test("edge_metadata_over_limit", test_edge_metadata_over_limit());
    runner.run_test("edge_max_metadata_size", test_edge_max_metadata_size());
    runner.run_test("edge_wrong_dimension", test_edge_wrong_dimension());
    runner.run_test("edge_empty_embedding", test_edge_empty_embedding());
    runner.run_test("edge_nan_embedding", test_edge_nan_embedding());
    runner.run_test("edge_inf_embedding", test_edge_inf_embedding());
    runner.run_test("edge_negative_doc_id", test_edge_negative_doc_id());
    runner.run_test("edge_empty_content_and_metadata", test_edge_empty_content_and_metadata());
    runner.run_test("edge_unicode_content", test_edge_unicode_content());

    runner.print_summary();

    run_benchmarks(embedding_dim, num_documents);

    return runner.all_passed() ? 0 : 1;
}

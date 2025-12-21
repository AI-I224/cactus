#include "../cactus/index/index.h"
#include "test_utils.h"
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

std::vector<float> random_embedding(size_t dim) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> embedding(dim);
    for (auto& val : embedding) {
        val = dist(gen);
    }
    return embedding;
}

void create_test_index(const std::string& index_path, const std::string& data_path, size_t embedding_dim, uint32_t num_docs = 0) {
    std::ofstream index_file(index_path, std::ios::binary);
    uint32_t magic = cactus::index::MAGIC;
    uint32_t version = cactus::index::VERSION;
    uint64_t embedding_dim_64 = static_cast<uint64_t>(embedding_dim);
    uint64_t num_docs_64 = static_cast<uint64_t>(num_docs);
    index_file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    index_file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    index_file.write(reinterpret_cast<const char*>(&embedding_dim_64), sizeof(uint64_t));
    index_file.write(reinterpret_cast<const char*>(&num_docs_64), sizeof(uint64_t));
    index_file.close();

    std::ofstream data_file(data_path, std::ios::binary);
    data_file.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
    data_file.write(reinterpret_cast<const char*>(&version), sizeof(uint32_t));
    data_file.close();
}

void cleanup_test_files(const std::string& index_path, const std::string& data_path) {
    unlink(index_path.c_str());
    unlink(data_path.c_str());
}

template<typename TestFunc>
bool run_isolated_test(const std::string& test_name, size_t embedding_dim, TestFunc test_func) {
    const std::string index_path = "/tmp/test_" + test_name + ".idx";
    const std::string data_path = "/tmp/test_" + test_name + ".dat";

    cleanup_test_files(index_path, data_path);

    try {
        create_test_index(index_path, data_path, embedding_dim);
        bool result = test_func(index_path, data_path, embedding_dim);
        cleanup_test_files(index_path, data_path);
        return result;
    } catch (...) {
        cleanup_test_files(index_path, data_path);
        return false;
    }
}

bool expect_exception(std::function<void()> func) {
    try {
        func();
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

bool test_constructor_basic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);
        return true;
    };
    return run_isolated_test("constructor_basic", embedding_dim, test);
}

bool test_constructor_missing_index(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        unlink(index_path.c_str());
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_missing_index", embedding_dim, test);
}

bool test_constructor_missing_data(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        unlink(data_path.c_str());
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_missing_data", embedding_dim, test);
}

bool test_constructor_wrong_magic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        std::ofstream index_file(index_path, std::ios::binary);
        uint32_t wrong_magic = 0xDEADBEEF;
        index_file.write(reinterpret_cast<const char*>(&wrong_magic), sizeof(uint32_t));
        index_file.close();

        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, dim);
        });
    };
    return run_isolated_test("constructor_wrong_magic", embedding_dim, test);
}

bool test_constructor_dimension_mismatch(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, uint32_t) {
        return expect_exception([&]() {
            cactus::index::Index index(index_path, data_path, 256);
        });
    };
    return run_isolated_test("constructor_dimension_mismatch", embedding_dim, test);
}

bool test_add_single_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content1", "metadata1"}
        };

        index.add_documents(docs);

        auto retrieved = index.get_documents({1});
        return retrieved.size() == 1 && retrieved[0].id == 1;
    };
    return run_isolated_test("add_single_document", embedding_dim, test);
}

bool test_add_multiple_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (uint32_t i = 0; i < 10; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "metadata" + std::to_string(i)
            });
        }

        index.add_documents(docs);

        auto retrieved = index.get_documents({0, 5, 9});
        return retrieved.size() == 3 &&
               retrieved[0].id == 0 &&
               retrieved[1].id == 5 &&
               retrieved[2].id == 9;
    };
    return run_isolated_test("add_multiple_documents", embedding_dim, test);
}

bool test_add_empty_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);
        std::vector<cactus::index::Document> docs;

        return expect_exception([&]() {
            index.add_documents(docs);
        });
    };
    return run_isolated_test("add_empty_documents", embedding_dim, test);
}

bool test_add_large_content(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::string large_content(1024 * 1024, 'x');
        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), large_content, "metadata1"}
        };

        return expect_exception([&]() {
            index.add_documents(docs);
        });
    };
    return run_isolated_test("add_large_content", embedding_dim, test);
}

bool test_delete_single_document(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs = {
            {1, random_embedding(dim), "content1", "metadata1"},
            {2, random_embedding(dim), "content2", "metadata2"}
        };
        index.add_documents(docs);
        index.delete_documents({1});

        return expect_exception([&]() {
            index.get_documents({1});
        });
    };
    return run_isolated_test("delete_single_document", embedding_dim, test);
}

bool test_query_basic(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        auto embedding1 = random_embedding(dim);
        auto embedding2 = random_embedding(dim);

        std::vector<cactus::index::Document> docs = {
            {1, embedding1, "content1", "metadata1"},
            {2, embedding2, "content2", "metadata2"}
        };
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 1;

        auto results = index.query({embedding1}, options);
        return results.size() == 1 && results[0].size() == 1 && results[0][0].doc_id == 1;
    };
    return run_isolated_test("query_basic", embedding_dim, test);
}

bool test_query_topk(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (uint32_t i = 0; i < 10; ++i) {
            docs.push_back({i, random_embedding(dim), "content", "meta"});
        }
        index.add_documents(docs);

        cactus::index::SearchOptions options;
        options.top_k = 5;

        auto results = index.query({random_embedding(dim)}, options);
        return results.size() == 1 && results[0].size() == 5;
    };
    return run_isolated_test("query_topk", embedding_dim, test);
}

bool test_stress_many_documents(size_t embedding_dim) {
    auto test = [](const std::string& index_path, const std::string& data_path, size_t dim) {
        cactus::index::Index index(index_path, data_path, dim);

        std::vector<cactus::index::Document> docs;
        for (uint32_t i = 0; i < 1000; ++i) {
            docs.push_back({
                i,
                random_embedding(dim),
                "content" + std::to_string(i),
                "meta" + std::to_string(i)
            });
        }

        index.add_documents(docs);

        auto retrieved = index.get_documents({500});
        return retrieved.size() == 1 && retrieved[0].id == 500;
    };
    return run_isolated_test("stress_many_documents", embedding_dim, test);
}

void run_benchmarks(size_t embedding_dim, uint32_t num_docs) {
    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Index Benchmark Suite                ║\n"
              << "║     Documents: " << std::setw(26) << std::left << num_docs << "║\n"
              << "╚══════════════════════════════════════════╝\n";

    const std::string index_path = "/tmp/bench_index.idx";
    const std::string data_path = "/tmp/bench_data.dat";

    // ========================================
    // INITIALIZATION: Populate index with data
    // ========================================
    std::cout << "\n[INITIALIZATION: Creating and populating index]\n";
    create_test_index(index_path, data_path, embedding_dim);

    auto start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index(index_path, data_path, embedding_dim);
    auto end = std::chrono::high_resolution_clock::now();
    auto init_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "├─ Empty index creation: " << init_duration << "ms\n";

    std::vector<cactus::index::Document> docs;
    docs.reserve(num_docs);
    for (uint32_t i = 0; i < num_docs; ++i) {
        docs.push_back({
            i,
            random_embedding(embedding_dim),
            "content_" + std::to_string(i),
            "meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(docs);
    end = std::chrono::high_resolution_clock::now();
    auto populate_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "├─ Populate " << num_docs << " docs: " << populate_duration << "ms\n"
              << "└─ Total initialization: " << (init_duration + populate_duration) << "ms\n";

    // ========================================
    // BENCHMARK: Init (load existing index)
    // ========================================
    std::cout << "\n[BENCHMARK: Init - Load existing index]\n";

    start = std::chrono::high_resolution_clock::now();
    cactus::index::Index index2(index_path, data_path, embedding_dim);
    end = std::chrono::high_resolution_clock::now();
    auto load_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << load_duration << "ms\n";

    // ========================================
    // BENCHMARK: Add (incremental adds)
    // ========================================
    std::cout << "\n[BENCHMARK: Add - Incremental document addition]\n";

    const uint32_t num_adds = std::min(1000u, num_docs / 10);
    std::vector<cactus::index::Document> new_docs;
    new_docs.reserve(num_adds);
    for (uint32_t i = 0; i < num_adds; ++i) {
        new_docs.push_back({
            num_docs + i,
            random_embedding(embedding_dim),
            "new_content_" + std::to_string(i),
            "new_meta_" + std::to_string(i)
        });
    }

    start = std::chrono::high_resolution_clock::now();
    index.add_documents(new_docs);
    end = std::chrono::high_resolution_clock::now();
    auto add_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << add_duration << "ms\n";

    // ========================================
    // BENCHMARK: Query (similarity search)
    // ========================================
    std::cout << "\n[BENCHMARK: Query - Similarity search]\n";
    cactus::index::SearchOptions options;
    options.top_k = 10;

    auto query_embedding = random_embedding(embedding_dim);

    start = std::chrono::high_resolution_clock::now();
    index.query({query_embedding}, options);
    end = std::chrono::high_resolution_clock::now();
    auto query_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << query_duration << "ms\n";

    // ========================================
    // BENCHMARK: Get (retrieve by ID)
    // ========================================
    std::cout << "\n[BENCHMARK: Get - Retrieve documents by ID]\n";

    const int num_gets = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<uint32_t> get_doc_ids;
    get_doc_ids.reserve(num_gets);
    for (int i = 0; i < num_gets; ++i) {
        get_doc_ids.push_back(i * (num_docs / num_gets));
    }

    start = std::chrono::high_resolution_clock::now();
    for (const auto& doc_id : get_doc_ids) {
        index.get_documents({doc_id});
    }
    end = std::chrono::high_resolution_clock::now();
    auto get_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << get_duration << "ms\n";

    // ========================================
    // BENCHMARK: Delete
    // ========================================
    std::cout << "\n[BENCHMARK: Delete - Remove documents]\n";

    const int num_deletes = std::min(1000, static_cast<int>(num_docs / 10));
    std::vector<uint32_t> delete_doc_ids;
    delete_doc_ids.reserve(num_deletes);
    for (int i = 0; i < num_deletes; ++i) {
        delete_doc_ids.push_back(i);
    }

    start = std::chrono::high_resolution_clock::now();
    for (const auto& doc_id : delete_doc_ids) {
        index.delete_documents({doc_id});
    }
    end = std::chrono::high_resolution_clock::now();
    auto delete_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "└─ Time: " << delete_duration << "ms\n";

    std::cout << "\n╔══════════════════════════════════════════╗\n"
              << "║     Benchmark Summary                    ║\n"
              << "╠══════════════════════════════════════════╣\n"
              << "║ Init:   " << std::setw(30) << std::right << load_duration << "ms ║\n"
              << "║ Add:    " << std::setw(30) << std::right << add_duration << "ms ║\n"
              << "║ Query:  " << std::setw(30) << std::right << query_duration << "ms ║\n"
              << "║ Get:    " << std::setw(30) << std::right << get_duration << "ms ║\n"
              << "║ Delete: " << std::setw(30) << std::right << delete_duration << "ms ║\n"
              << "╚══════════════════════════════════════════╝\n";

    cleanup_test_files(index_path, data_path);
}

int main() {
    const size_t embedding_dim = 1024;
    const uint32_t num_documents = 1000000;

    TestUtils::TestRunner runner("Index Tests");

    runner.run_test("constructor_basic", test_constructor_basic(embedding_dim));
    runner.run_test("constructor_missing_index", test_constructor_missing_index(embedding_dim));
    runner.run_test("constructor_missing_data", test_constructor_missing_data(embedding_dim));
    runner.run_test("constructor_wrong_magic", test_constructor_wrong_magic(embedding_dim));
    runner.run_test("constructor_dimension_mismatch", test_constructor_dimension_mismatch(embedding_dim));
    runner.run_test("add_single_document", test_add_single_document(embedding_dim));
    runner.run_test("add_multiple_documents", test_add_multiple_documents(embedding_dim));
    runner.run_test("add_empty_documents", test_add_empty_documents(embedding_dim));
    runner.run_test("add_large_content", test_add_large_content(embedding_dim));
    runner.run_test("delete_single_document", test_delete_single_document(embedding_dim));
    runner.run_test("query_basic", test_query_basic(embedding_dim));
    runner.run_test("query_topk", test_query_topk(embedding_dim));
    runner.run_test("stress_many_documents", test_stress_many_documents(embedding_dim));

    runner.print_summary();

    run_benchmarks(embedding_dim, num_documents);

    return runner.all_passed() ? 0 : 1;
}

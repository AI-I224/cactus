#include "test_utils.h"
#include "../cactus/kernel/kernel_utils.h"
#include <chrono>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <limits>
#include <map>

struct BenchResult {
    size_t M, K, N;
    size_t total_tiles;
    size_t num_threads;
    double avg_time_ms;
    double gflops;
};

std::vector<BenchResult> benchmark_gemm_threading() {
    const std::vector<size_t> M_values = {1, 8, 64, 128, 256, 1024};
    const std::vector<size_t> K_values = {256, 512, 1024};
    const std::vector<size_t> N_values = {1024, 2048};
    const int iterations = 5;
    const size_t group_size = 128;

    auto& pool = CactusThreading::get_thread_pool();
    size_t max_threads = pool.num_workers();
    std::vector<size_t> thread_counts;
    for (size_t t = 1; t <= max_threads; t++) {
        thread_counts.push_back(t);
    }

    #if defined(__APPLE__) && defined(__arm64__)
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 8;
    #else
    constexpr size_t TILE_M = 4;
    constexpr size_t TILE_N = 4;
    #endif

    std::mt19937 gen(42);
    std::uniform_real_distribution<float> float_dist(-1.0f, 1.0f);
    std::uniform_int_distribution<int> int_dist(-127, 127);

    std::vector<BenchResult> results;
    size_t total_dims = M_values.size() * K_values.size() * N_values.size();
    size_t dim_idx = 0;

    std::cout << "\n";
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│              GEMM INT8 M-Based Threading Benchmark                              │\n";
    std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Testing thread counts: 1 to " << max_threads << std::setw(51) << "│\n";
    std::cout << "│ Dimensions: " << total_dims << " combos, " << iterations << " iterations each"
              << std::setw(40) << "│\n";
    std::cout << "│ Tile size: " << TILE_M << "x" << TILE_N
              << std::setw(60) << "│\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘\n";
    std::cout << "\n";

    for (size_t M : M_values) {
        for (size_t K : K_values) {
            for (size_t N : N_values) {
                dim_idx++;
                size_t K_aligned = ((K + group_size - 1) / group_size) * group_size;
                size_t num_groups = K_aligned / group_size;

                size_t num_row_tiles = (M + TILE_M - 1) / TILE_M;
                size_t num_col_tiles = (N + TILE_N - 1) / TILE_N;
                size_t total_tiles = num_row_tiles * num_col_tiles;

                std::vector<__fp16> A(M * K_aligned);
                std::vector<int8_t> B(N * K_aligned);
                std::vector<__fp16> B_scales(num_groups * N);
                std::vector<__fp16> C(M * N);

                for (size_t i = 0; i < M * K_aligned; ++i) {
                    A[i] = static_cast<__fp16>(float_dist(gen));
                }
                for (size_t i = 0; i < N * K_aligned; ++i) {
                    B[i] = static_cast<int8_t>(int_dist(gen));
                }
                for (size_t i = 0; i < num_groups * N; ++i) {
                    B_scales[i] = static_cast<__fp16>(0.01f + std::abs(float_dist(gen)) * 0.05f);
                }

                BenchResult best = {M, K, N, total_tiles, 0, std::numeric_limits<double>::max(), 0};

                std::cout << "[" << dim_idx << "/" << total_dims << "] "
                          << "M=" << std::setw(4) << M
                          << " K=" << std::setw(4) << K
                          << " N=" << std::setw(4) << N
                          << " (tiles=" << std::setw(5) << total_tiles << ") ... " << std::flush;

                for (size_t num_threads : thread_counts) {
                    CactusThreading::set_gemm_threads(num_threads);

                    // Warmup
                    cactus_matmul_int(A.data(), B.data(), B_scales.data(), C.data(),
                                     M, K_aligned, N, group_size);

                    // Benchmark
                    auto start = std::chrono::high_resolution_clock::now();
                    for (int i = 0; i < iterations; ++i) {
                        cactus_matmul_int(A.data(), B.data(), B_scales.data(), C.data(),
                                         M, K_aligned, N, group_size);
                    }
                    auto end = std::chrono::high_resolution_clock::now();
                    double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

                    if (avg_ms < best.avg_time_ms) {
                        best.num_threads = num_threads;
                        best.avg_time_ms = avg_ms;
                    }
                }

                best.gflops = (2.0 * M * K_aligned * N) / (best.avg_time_ms * 1e6);

                std::cout << "threads=" << std::setw(2) << best.num_threads
                          << " -> " << std::fixed << std::setprecision(3) << best.avg_time_ms << "ms"
                          << " (" << std::setprecision(1) << best.gflops << " GFLOPS)\n";

                results.push_back(best);
            }
        }
    }

    CactusThreading::reset_gemm_threads();
    return results;
}

void print_results_table(const std::vector<BenchResult>& results) {
    std::cout << "\n";
    std::cout << "┌────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                          OPTIMAL THREAD COUNTS                                 │\n";
    std::cout << "├──────┬──────┬──────┬────────┬─────────┬───────────┬──────────────────────────┤\n";
    std::cout << "│  M   │  K   │  N   │ Tiles  │ Threads │ Time(ms)  │ GFLOPS                   │\n";
    std::cout << "├──────┼──────┼──────┼────────┼─────────┼───────────┼──────────────────────────┤\n";

    for (const auto& r : results) {
        std::cout << "│" << std::setw(5) << r.M << " "
                  << "│" << std::setw(5) << r.K << " "
                  << "│" << std::setw(5) << r.N << " "
                  << "│" << std::setw(7) << r.total_tiles << " "
                  << "│" << std::setw(8) << r.num_threads << " "
                  << "│" << std::setw(10) << std::fixed << std::setprecision(3) << r.avg_time_ms << " "
                  << "│" << std::setw(10) << std::setprecision(1) << r.gflops << "                │\n";
    }

    std::cout << "└──────┴──────┴──────┴────────┴─────────┴───────────┴──────────────────────────┘\n";

    std::cout << "\n";
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    RECOMMENDED THREAD COUNTS BY BATCH SIZE                      │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘\n";

    std::map<size_t, std::vector<BenchResult>> by_batch;
    for (const auto& r : results) {
        by_batch[r.M].push_back(r);
    }

    std::cout << "\n  GemmThreading::get_num_threads(M, pool_size) should return:\n\n";

    for (const auto& [M, batch_results] : by_batch) {
        std::map<size_t, int> thread_freq;
        double total_gflops = 0;
        for (const auto& r : batch_results) {
            thread_freq[r.num_threads]++;
            total_gflops += r.gflops;
        }

        size_t best_threads = 0;
        int max_freq = 0;
        for (const auto& [t, f] : thread_freq) {
            if (f > max_freq) { max_freq = f; best_threads = t; }
        }

        std::string workload = (M <= 1) ? "decode" : (M <= 8) ? "small" : (M <= 64) ? "medium" : "prefill";
        std::cout << "    M=" << std::setw(4) << M << " (" << std::setw(6) << workload << "): "
                  << std::setw(2) << best_threads << " threads"
                  << "  (avg " << std::fixed << std::setprecision(1)
                  << (total_gflops / batch_results.size()) << " GFLOPS)\n";
    }

    std::cout << "\n";
}

int main() {
    auto results = benchmark_gemm_threading();
    print_results_table(results);
    return 0;
}

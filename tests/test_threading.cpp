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
    size_t min_work_gate;
    size_t work_per_thread;
    double avg_time_ms;
    double gflops;
};

std::vector<BenchResult> benchmark_gemm_threading() {
    const std::vector<size_t> M_values = {1, 128, 1024};
    const std::vector<size_t> K_values = {256, 512, 1024};
    const std::vector<size_t> N_values = {1024, 2048};
    const std::vector<size_t> search_space = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    const int iterations = 5;
    const size_t group_size = 128;

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
    std::cout << "│                    GEMM INT8 Groupwise Threading Benchmark                      │\n";
    std::cout << "├─────────────────────────────────────────────────────────────────────────────────┤\n";
    std::cout << "│ Search space: " << search_space.size() << "x" << search_space.size()
              << " = " << (search_space.size() * search_space.size()) << " configs per dimension"
              << std::setw(28) << "│\n";
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

                BenchResult best = {M, K, N, total_tiles, 0, 0, std::numeric_limits<double>::max(), 0};

                std::cout << "[" << dim_idx << "/" << total_dims << "] "
                          << "M=" << std::setw(4) << M
                          << " K=" << std::setw(4) << K
                          << " N=" << std::setw(4) << N
                          << " (tiles=" << std::setw(5) << total_tiles << ") ... " << std::flush;

                for (size_t gate : search_space) {
                    for (size_t per_thread : search_space) {
                        CactusThreading::set_gemm_config(gate, per_thread);

                        cactus_matmul_int(A.data(), B.data(), B_scales.data(), C.data(),
                                         M, K_aligned, N, group_size);

                        auto start = std::chrono::high_resolution_clock::now();
                        for (int i = 0; i < iterations; ++i) {
                            cactus_matmul_int(A.data(), B.data(), B_scales.data(), C.data(),
                                             M, K_aligned, N, group_size);
                        }
                        auto end = std::chrono::high_resolution_clock::now();
                        double avg_ms = std::chrono::duration<double, std::milli>(end - start).count() / iterations;

                        if (avg_ms < best.avg_time_ms) {
                            best.min_work_gate = gate;
                            best.work_per_thread = per_thread;
                            best.avg_time_ms = avg_ms;
                        }
                    }
                }

                best.gflops = (2.0 * M * K_aligned * N) / (best.avg_time_ms * 1e6);

                std::cout << "gate=" << std::setw(5) << best.min_work_gate
                          << " per_thread=" << std::setw(5) << best.work_per_thread
                          << " -> " << std::fixed << std::setprecision(3) << best.avg_time_ms << "ms"
                          << " (" << std::setprecision(1) << best.gflops << " GFLOPS)\n";

                results.push_back(best);
            }
        }
    }

    CactusThreading::reset_gemm_config();
    return results;
}

void print_results_table(const std::vector<BenchResult>& results) {
    std::cout << "\n";
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                               OPTIMAL CONFIGS SUMMARY                                       │\n";
    std::cout << "├──────┬──────┬──────┬────────┬────────────┬──────────────┬───────────┬───────────────────────┤\n";
    std::cout << "│  M   │  K   │  N   │ Tiles  │ min_gate   │ per_thread   │ Time(ms)  │ GFLOPS                │\n";
    std::cout << "├──────┼──────┼──────┼────────┼────────────┼──────────────┼───────────┼───────────────────────┤\n";

    for (const auto& r : results) {
        std::cout << "│" << std::setw(5) << r.M << " "
                  << "│" << std::setw(5) << r.K << " "
                  << "│" << std::setw(5) << r.N << " "
                  << "│" << std::setw(7) << r.total_tiles << " "
                  << "│" << std::setw(11) << r.min_work_gate << " "
                  << "│" << std::setw(13) << r.work_per_thread << " "
                  << "│" << std::setw(10) << std::fixed << std::setprecision(3) << r.avg_time_ms << " "
                  << "│" << std::setw(10) << std::setprecision(1) << r.gflops << "             │\n";
    }

    std::cout << "└──────┴──────┴──────┴────────┴────────────┴──────────────┴───────────┴───────────────────────┘\n";

    std::cout << "\n";
    std::cout << "┌─────────────────────────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                           ANALYSIS BY BATCH SIZE                                │\n";
    std::cout << "└─────────────────────────────────────────────────────────────────────────────────┘\n";

    std::map<size_t, std::vector<BenchResult>> by_batch;
    for (const auto& r : results) {
        by_batch[r.M].push_back(r);
    }

    for (const auto& [M, batch_results] : by_batch) {
        std::map<size_t, int> gate_freq, per_thread_freq;
        double total_gflops = 0;
        for (const auto& r : batch_results) {
            gate_freq[r.min_work_gate]++;
            per_thread_freq[r.work_per_thread]++;
            total_gflops += r.gflops;
        }

        size_t best_gate = 0, best_per = 0;
        int max_gate_freq = 0, max_per_freq = 0;
        for (const auto& [g, f] : gate_freq) {
            if (f > max_gate_freq) { max_gate_freq = f; best_gate = g; }
        }
        for (const auto& [p, f] : per_thread_freq) {
            if (f > max_per_freq) { max_per_freq = f; best_per = p; }
        }

        std::cout << "\n  M=" << M << " (batch=" << (M == 1 ? "decode" : "prefill") << "):\n";
        std::cout << "    Recommended: ParallelConfig{" << best_gate << ", " << best_per << "}\n";
        std::cout << "    Avg GFLOPS: " << std::fixed << std::setprecision(1)
                  << (total_gflops / batch_results.size()) << "\n";
    }

    std::cout << "\n";
}

int main() {
    std::cout << "\n";

    auto results = benchmark_gemm_threading();
    print_results_table(results);

    return 0;
}

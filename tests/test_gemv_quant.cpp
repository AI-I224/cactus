/**
 * GEMV FP4 Weight-Only Quantization Tests
 *
 * Benchmarks FP4 quantization for GEMV (M=1):
 * - Tests the cactus_matmul_f4_to_int32 kernel with cactus_int32_to_fp16_scaled conversion
 * - Compares against FP16 and INT8 baselines
 */

#include "test_utils.h"
#include <vector>
#include <cmath>
#include <random>
#include <chrono>

//==============================================================================
// Benchmark Utilities
//==============================================================================

template<typename F>
double measure_time_ms(F func, int warmup = 3, int iterations = 10) {
    for (int i = 0; i < warmup; ++i) func();

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count() / iterations;
}

//==============================================================================
// FP4 E2M1 Format
//==============================================================================
// E2M1: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (and negatives)
// Exponent bias = 1
//
// 4-bit encoding: [sign(1)][exp(2)][mantissa(1)]
//   0b0000 = +0.0    0b1000 = -0.0
//   0b0001 = +0.5    0b1001 = -0.5   (subnormal: 0 * 2^0 + 0.5)
//   0b0010 = +1.0    0b1010 = -1.0   (1.0 * 2^0)
//   0b0011 = +1.5    0b1011 = -1.5   (1.5 * 2^0)
//   0b0100 = +2.0    0b1100 = -2.0   (1.0 * 2^1)
//   0b0101 = +3.0    0b1101 = -3.0   (1.5 * 2^1)
//   0b0110 = +4.0    0b1110 = -4.0   (1.0 * 2^2)
//   0b0111 = +6.0    0b1111 = -6.0   (1.5 * 2^2)

// LUT: 4-bit E2M1 code -> float value (for codes 0-7, positive values)
static const float fp4_e2m1_to_float[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Quantize float to 4-bit E2M1 code (0-15), assumes scale=1
inline uint8_t float_to_fp4_e2m1(float val) {
    uint8_t sign = (val < 0) ? 0x8 : 0x0;
    float abs_val = std::abs(val);

    // Find nearest E2M1 value (round half up - ties go to larger magnitude)
    uint8_t best_code = 0;
    float best_dist = abs_val;  // distance to 0
    for (uint8_t i = 1; i < 8; ++i) {
        float dist = std::abs(abs_val - fp4_e2m1_to_float[i]);
        // Use <= for "round half up" behavior (ties go to larger value)
        if (dist < best_dist || (dist == best_dist && i > best_code)) {
            best_dist = dist;
            best_code = i;
        }
    }

    return sign | best_code;
}

// Dequantize 4-bit E2M1 code to float, assumes scale=1
inline float fp4_e2m1_to_float_val(uint8_t code) {
    uint8_t sign = (code >> 3) & 0x1;
    uint8_t magnitude = code & 0x7;
    float val = fp4_e2m1_to_float[magnitude];
    return sign ? -val : val;
}

// Pack two FP4 E2M1 values into one byte: [val1(4bits) | val0(4bits)]
// val0 in low nibble, val1 in high nibble (matches kernel's load order)
inline uint8_t pack_fp4_pair(uint8_t fp4_0, uint8_t fp4_1) {
    return (fp4_1 << 4) | (fp4_0 & 0x0F);
}

// Unpack two FP4 E2M1 values from one byte
inline void unpack_fp4_pair(uint8_t packed, uint8_t& fp4_0, uint8_t& fp4_1) {
    fp4_0 = packed & 0x0F;           // val0 from low nibble
    fp4_1 = (packed >> 4) & 0x0F;    // val1 from high nibble
}

// Pack an array of floats into FP4 E2M1 packed format (2 values per byte)
// Input: float array of size n (must be even)
// Output: uint8_t array of size n/2
inline void pack_fp4_array(const float* input, uint8_t* output, size_t n) {
    for (size_t i = 0; i < n; i += 2) {
        uint8_t fp4_0 = float_to_fp4_e2m1(input[i]);
        uint8_t fp4_1 = float_to_fp4_e2m1(input[i + 1]);
        output[i / 2] = pack_fp4_pair(fp4_0, fp4_1);
    }
}

// Unpack FP4 E2M1 packed array back to floats
inline void unpack_fp4_array(const uint8_t* input, float* output, size_t n) {
    for (size_t i = 0; i < n / 2; ++i) {
        uint8_t fp4_0, fp4_1;
        unpack_fp4_pair(input[i], fp4_0, fp4_1);
        output[i * 2] = fp4_e2m1_to_float_val(fp4_0);
        output[i * 2 + 1] = fp4_e2m1_to_float_val(fp4_1);
    }
}

//==============================================================================
// Correctness Tests
//==============================================================================

bool test_fp4_e2m1_pack_unpack() {
    // Test that packing and unpacking FP4 E2M1 values is lossless

    // All 16 possible FP4 E2M1 values
    float test_values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,    // positive
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f  // negative
    };

    // Test individual quantization/dequantization
    int failures = 0;
    for (int i = 0; i < 16; ++i) {
        float val = test_values[i];
        uint8_t code = float_to_fp4_e2m1(val);
        float recovered = fp4_e2m1_to_float_val(code);

        // For -0.0, we expect +0.0 back (both are valid representations of zero)
        float expected = (val == -0.0f) ? 0.0f : val;

        if (std::abs(recovered - expected) > 1e-6f) {
            printf("  Single value mismatch: input=%.2f, code=0x%X, recovered=%.2f\n",
                   val, code, recovered);
            failures++;
        }
    }

    // Test pack/unpack roundtrip for pairs
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            float val0 = fp4_e2m1_to_float[i];
            float val1 = fp4_e2m1_to_float[j];

            // Test positive pair
            uint8_t code0 = float_to_fp4_e2m1(val0);
            uint8_t code1 = float_to_fp4_e2m1(val1);
            uint8_t packed = pack_fp4_pair(code0, code1);

            uint8_t unpacked0, unpacked1;
            unpack_fp4_pair(packed, unpacked0, unpacked1);

            if (unpacked0 != code0 || unpacked1 != code1) {
                printf("  Pack/unpack mismatch: codes=(%d,%d), packed=0x%02X, unpacked=(%d,%d)\n",
                       code0, code1, packed, unpacked0, unpacked1);
                failures++;
            }

            // Test negative pair
            float neg_val0 = -val0;
            float neg_val1 = -val1;
            uint8_t neg_code0 = float_to_fp4_e2m1(neg_val0);
            uint8_t neg_code1 = float_to_fp4_e2m1(neg_val1);
            uint8_t neg_packed = pack_fp4_pair(neg_code0, neg_code1);

            uint8_t neg_unpacked0, neg_unpacked1;
            unpack_fp4_pair(neg_packed, neg_unpacked0, neg_unpacked1);

            if (neg_unpacked0 != neg_code0 || neg_unpacked1 != neg_code1) {
                printf("  Neg pack/unpack mismatch: codes=(%d,%d), packed=0x%02X, unpacked=(%d,%d)\n",
                       neg_code0, neg_code1, neg_packed, neg_unpacked0, neg_unpacked1);
                failures++;
            }
        }
    }

    // Test array pack/unpack
    const size_t n = 32;
    std::vector<float> input(n);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> idx_dist(0, 15);

    for (size_t i = 0; i < n; ++i) {
        input[i] = test_values[idx_dist(rng)];
    }

    std::vector<uint8_t> packed(n / 2);
    pack_fp4_array(input.data(), packed.data(), n);

    std::vector<float> unpacked(n);
    unpack_fp4_array(packed.data(), unpacked.data(), n);

    for (size_t i = 0; i < n; ++i) {
        float expected = (input[i] == -0.0f) ? 0.0f : input[i];
        if (std::abs(unpacked[i] - expected) > 1e-6f) {
            printf("  Array mismatch at %zu: input=%.2f, unpacked=%.2f\n",
                   i, input[i], unpacked[i]);
            failures++;
        }
    }

    // Test quantization of values between E2M1 grid points (should snap to nearest)
    struct SnapTest { float input; float expected; };
    SnapTest snap_tests[] = {
        {0.25f, 0.5f},   // Between 0 and 0.5, closer to 0.5
        {0.24f, 0.0f},   // Between 0 and 0.5, closer to 0
        {0.75f, 1.0f},   // Between 0.5 and 1.0, closer to 1.0
        {1.25f, 1.5f},   // Between 1.0 and 1.5, closer to 1.5
        {1.74f, 1.5f},   // Between 1.5 and 2.0, closer to 1.5
        {1.76f, 2.0f},   // Between 1.5 and 2.0, closer to 2.0
        {2.5f, 3.0f},    // Between 2.0 and 3.0, closer to 3.0
        {3.5f, 4.0f},    // Between 3.0 and 4.0, closer to 4.0
        {5.0f, 6.0f},    // Between 4.0 and 6.0, closer to 6.0
        {4.9f, 4.0f},    // Between 4.0 and 6.0, closer to 4.0
        {7.0f, 6.0f},    // Beyond max, should clamp to 6.0
        {-2.5f, -3.0f},  // Negative, between -2.0 and -3.0
    };

    for (const auto& test : snap_tests) {
        uint8_t code = float_to_fp4_e2m1(test.input);
        float recovered = fp4_e2m1_to_float_val(code);
        if (std::abs(recovered - test.expected) > 1e-6f) {
            printf("  Snap test failed: input=%.2f, expected=%.2f, got=%.2f\n",
                   test.input, test.expected, recovered);
            failures++;
        }
    }

    printf("  E2M1 pack/unpack tests: %d failures\n", failures);
    return failures == 0;
}

bool test_fp4_kernel_runs() {
    // Basic test: verify the kernel runs without crashing
    const size_t M = 1, K = 64, N = 32;

    std::vector<int8_t> a(M * K);
    std::vector<int8_t> b(N * K);
    std::vector<int32_t> c_i32(M * N);
    std::vector<__fp16> c(M * N);

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-7, 7);

    for (size_t i = 0; i < M * K; ++i) a[i] = static_cast<int8_t>(dist(rng));
    for (size_t i = 0; i < N * K; ++i) b[i] = static_cast<int8_t>(dist(rng));

    cactus_matmul_f4_to_int32(a.data(), b.data(), c_i32.data(), M, K, N);
    cactus_int32_to_fp16_scaled(c_i32.data(), c.data(), M * N, 0.25f);

    // Just verify we got some output
    bool has_nonzero = false;
    for (size_t i = 0; i < M * N; ++i) {
        if (c[i] != 0) has_nonzero = true;
    }

    printf("  FP4 kernel executed successfully\n");
    return has_nonzero;
}

bool test_fp4_larger_sizes() {
    // Test with larger sizes to exercise parallel paths
    const size_t M = 1, K = 1024, N = 512;

    std::vector<int8_t> a(M * K);
    std::vector<int8_t> b(N * K);
    std::vector<int32_t> c_i32(M * N);
    std::vector<__fp16> c(M * N);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(-7, 7);

    for (size_t i = 0; i < M * K; ++i) a[i] = static_cast<int8_t>(dist(rng));
    for (size_t i = 0; i < N * K; ++i) b[i] = static_cast<int8_t>(dist(rng));

    cactus_matmul_f4_to_int32(a.data(), b.data(), c_i32.data(), M, K, N);
    cactus_int32_to_fp16_scaled(c_i32.data(), c.data(), M * N, 0.25f);

    bool has_nonzero = false;
    for (size_t i = 0; i < M * N; ++i) {
        if (c[i] != 0) has_nonzero = true;
    }

    printf("  FP4 kernel with larger sizes executed successfully\n");
    return has_nonzero;
}

bool test_fp4_correctness() {
    struct TestSize { size_t N; size_t K; const char* name; };
    std::vector<TestSize> sizes = {
        {1024, 1024, "1024x1024"},
        {4096, 4096, "4096x4096"},
        {4096, 11008, "4096x11008"},
        {11008, 4096, "11008x4096"},
    };

    const size_t M = 1;
    int total_failures = 0;

    // All 16 possible FP4 E2M1 values (for random selection)
    float e2m1_values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    for (const auto& sz : sizes) {
        const size_t N = sz.N;
        const size_t K = sz.K;

        const float tolerance = 0.25f;

        // Generate random data directly from E2M1 values (no scaling needed)
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> idx_dist(0, 15);

        std::vector<float> weights_f32(N * K);
        std::vector<float> input_f32(K);

        for (size_t i = 0; i < N * K; ++i) weights_f32[i] = e2m1_values[idx_dist(rng)];
        for (size_t i = 0; i < K; ++i) input_f32[i] = e2m1_values[idx_dist(rng)];

        // Quantize and pack weights to FP4 E2M1 (2 values per byte)
        std::vector<uint8_t> weights_fp4_packed(N * K / 2);
        for (size_t row = 0; row < N; ++row) {
            pack_fp4_array(&weights_f32[row * K], &weights_fp4_packed[row * K / 2], K);
        }

        // Quantize input to FP4 (same format as weights)
        std::vector<uint8_t> input_fp4_packed(K / 2);
        pack_fp4_array(input_f32.data(), input_fp4_packed.data(), K);

        // Compute reference using integer accumulation (same as kernel)
        // Kernel LUT: {0, 1, 2, 3, 4, 6, 8, 12} = 2x E2M1 values
        // Kernel computes: sum(lut_a * lut_b) / 4 = sum(e2m1_a * e2m1_b)
        // Then stores to fp16, so we also round to fp16 for exact comparison
        static const int8_t lut[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        std::vector<__fp16> output_ref(N);
        for (size_t n = 0; n < N; ++n) {
            int32_t sum = 0;
            for (size_t k = 0; k < K; k += 2) {
                uint8_t input_byte = input_fp4_packed[k / 2];
                uint8_t weight_byte = weights_fp4_packed[n * K / 2 + k / 2];

                int8_t a0 = lut[input_byte & 0x0F];
                int8_t a1 = lut[input_byte >> 4];
                int8_t b0 = lut[weight_byte & 0x0F];
                int8_t b1 = lut[weight_byte >> 4];

                sum += static_cast<int32_t>(a0) * static_cast<int32_t>(b0);
                sum += static_cast<int32_t>(a1) * static_cast<int32_t>(b1);
            }
            output_ref[n] = static_cast<__fp16>(sum / 4.0f);
        }

        // Compute FP4 result (using packed input and weights)
        std::vector<int32_t> output_i32_fp4(N);
        std::vector<__fp16> output_f16_fp4(N);
        cactus_matmul_f4_to_int32(reinterpret_cast<const int8_t*>(input_fp4_packed.data()),
                                  reinterpret_cast<const int8_t*>(weights_fp4_packed.data()),
                                  output_i32_fp4.data(), M, K, N);
        cactus_int32_to_fp16_scaled(output_i32_fp4.data(), output_f16_fp4.data(), N, 0.25f);

        // Compute errors and metrics
        std::vector<float> all_errors;
        float sum_err = 0.0f;
        float sum_sq_err = 0.0f;
        float dot_product = 0.0f;
        float norm_fp16_sq = 0.0f;
        float norm_fp4_sq = 0.0f;
        float max_fp16 = 0.0f;

        for (size_t i = 0; i < N; ++i) {
            float ref_val = static_cast<float>(output_ref[i]);
            float fp4_val = static_cast<float>(output_f16_fp4[i]);
            float err = std::abs(ref_val - fp4_val);

            all_errors.push_back(err);
            sum_err += err;
            sum_sq_err += err * err;
            dot_product += ref_val * fp4_val;
            norm_fp16_sq += ref_val * ref_val;
            norm_fp4_sq += fp4_val * fp4_val;
            max_fp16 = std::max(max_fp16, std::abs(ref_val));
        }
        std::sort(all_errors.begin(), all_errors.end());

        float max_err = all_errors.back();
        float mean_err = sum_err / static_cast<float>(N);
        float p50 = all_errors[N / 2];
        float p90 = all_errors[N * 90 / 100];
        float p99 = all_errors[N * 99 / 100];

        // MSE, CosSim, PSNR
        float mse = sum_sq_err / static_cast<float>(N);
        float cos_sim = dot_product / (std::sqrt(norm_fp16_sq) * std::sqrt(norm_fp4_sq) + 1e-8f);
        float psnr = (mse > 0) ? 10.0f * std::log10((max_fp16 * max_fp16) / mse) : INFINITY;

        // Count failures against tolerance
        int failures = 0;
        for (float err : all_errors) {
            if (err > tolerance) failures++;
        }

        // Print first 5 results on one line
        printf("  [%s] Ref: ", sz.name);
        for (size_t i = 0; i < 5 && i < N; ++i) {
            printf("%.2f ", static_cast<float>(output_ref[i]));
        }
        printf("| FP4: ");
        for (size_t i = 0; i < 5 && i < N; ++i) {
            printf("%.2f ", static_cast<float>(output_f16_fp4[i]));
        }
        printf("\n");

        printf("  [%s] err: max=%.4f, p99=%.4f, p90=%.4f, p50=%.4f, mean=%.4f\n",
               sz.name, max_err, p99, p90, p50, mean_err);
        printf("  [%s] MSE=%.6f, CosSim=%.6f, PSNR=%.2fdB, failures=%d/%zu\n",
               sz.name, mse, cos_sim, psnr, failures, N);

        total_failures += failures;
    }

    return total_failures == 0;
}

//==============================================================================
// Performance Benchmarks
//==============================================================================

bool benchmark_gemv_comparison(TestUtils::TestRunner& runner) {
    struct BenchSize { size_t N; size_t K; const char* name; };
    std::vector<BenchSize> sizes = {
        {1024, 1024, "1024x1024"},
        {4096, 4096, "4096x4096"},
        {4096, 11008, "4096x11008"},
        {11008, 4096, "11008x4096"},
    };

    // All 16 possible FP4 E2M1 values (for random selection)
    float e2m1_values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };

    for (const auto& sz : sizes) {
        const size_t N = sz.N;
        const size_t K = sz.K;
        const size_t M = 1;

        // Generate random data directly from E2M1 values
        std::mt19937 rng(42);
        std::uniform_int_distribution<int> idx_dist(0, 15);

        std::vector<float> weights_f32(N * K);
        std::vector<float> input_f32(K);

        for (size_t i = 0; i < N * K; ++i) weights_f32[i] = e2m1_values[idx_dist(rng)];
        for (size_t i = 0; i < K; ++i) input_f32[i] = e2m1_values[idx_dist(rng)];

        // Quantize and pack weights to FP4 E2M1 (2 values per byte)
        std::vector<uint8_t> weights_fp4_packed(N * K / 2);
        for (size_t row = 0; row < N; ++row) {
            pack_fp4_array(&weights_f32[row * K], &weights_fp4_packed[row * K / 2], K);
        }

        // Dequantize FP4 weights to FP16 for the FP16 baseline (same quantization loss)
        std::vector<__fp16> weights_f16_from_fp4(N * K);
        for (size_t row = 0; row < N; ++row) {
            std::vector<float> unpacked(K);
            unpack_fp4_array(&weights_fp4_packed[row * K / 2], unpacked.data(), K);
            for (size_t k = 0; k < K; ++k) {
                weights_f16_from_fp4[row * K + k] = static_cast<__fp16>(unpacked[k]);
            }
        }

        // INT8 scaling (for INT8 benchmark comparison)
        float max_weight = 6.0f;  // max E2M1 value
        float max_input = 6.0f;
        float int8_weight_scale = max_weight / 127.0f;
        float input_scale = max_input / 127.0f;

        // INT8 weights (full int8 range for comparison)
        std::vector<int8_t> weights_i8(N * K);
        for (size_t i = 0; i < N * K; ++i) {
            weights_i8[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, weights_f32[i] / int8_weight_scale))));
        }

        // Input quantization
        std::vector<__fp16> input_f16(K);
        std::vector<int8_t> input_i8(K);
        for (size_t i = 0; i < K; ++i) {
            input_f16[i] = static_cast<__fp16>(input_f32[i]);
            input_i8[i] = static_cast<int8_t>(std::round(std::max(-128.0f, std::min(127.0f, input_f32[i] / input_scale))));
        }

        std::vector<__fp16> output_f16(N);
        std::vector<int32_t> output_i32(N);

        size_t flops = 2 * M * N * K;
        char perf_str[256];

        // FP16 with FP4-quantized weights (dequantized)
        {
            double time = measure_time_ms([&]() {
                cactus_matmul_f16(input_f16.data(), weights_f16_from_fp4.data(), output_f16.data(), M, K, N);
            });
            double gflops = flops / (time * 1e6);
            snprintf(perf_str, sizeof(perf_str), "%.3f ms, %.2f GFLOPS", time, gflops);
            runner.log_performance(std::string("FP16 (") + sz.name + ")", perf_str);
        }

        // INT8
        {
            double time = measure_time_ms([&]() {
                cactus_matmul_int8_to_int32(input_i8.data(), weights_i8.data(), output_i32.data(), M, K, N);
            });
            double gflops = flops / (time * 1e6);
            snprintf(perf_str, sizeof(perf_str), "%.3f ms, %.2f GFLOPS", time, gflops);
            runner.log_performance(std::string("INT8 (") + sz.name + ")", perf_str);
        }

        // FP4 (using packed FP4 E2M1 weights - 2 values per byte)
        {
            std::vector<int32_t> output_i32_fp4(N);
            std::vector<__fp16> output_fp4(N);
            double time = measure_time_ms([&]() {
                cactus_matmul_f4_to_int32(input_i8.data(), reinterpret_cast<const int8_t*>(weights_fp4_packed.data()), output_i32_fp4.data(), M, K, N);
            });
            cactus_int32_to_fp16_scaled(output_i32_fp4.data(), output_fp4.data(), N, 0.25f);
            double gflops = flops / (time * 1e6);
            snprintf(perf_str, sizeof(perf_str), "%.3f ms, %.2f GFLOPS", time, gflops);
            runner.log_performance(std::string("FP4 (") + sz.name + ")", perf_str);
        }

        printf("\n");
    }

    return true;
}

//==============================================================================
// Main
//==============================================================================

int main() {
    TestUtils::TestRunner runner("GEMV FP4 Quantization Tests");

    printf("Hardware threads: %u\n\n", std::thread::hardware_concurrency());

    // Basic functionality tests
    runner.run_test("FP4 E2M1 Pack/Unpack", test_fp4_e2m1_pack_unpack());
    runner.run_test("FP4 Kernel Runs", test_fp4_kernel_runs());
    runner.run_test("FP4 Larger Sizes", test_fp4_larger_sizes());
    runner.run_test("FP4 Correctness", test_fp4_correctness());

    // Performance benchmarks
    printf("\n=== Performance Benchmarks (M=1 GEMV) ===\n");
    printf("Memory per weight: FP16=2B, INT8=1B, FP4=0.5B\n\n");

    benchmark_gemv_comparison(runner);

    runner.print_summary();

    return runner.all_passed() ? 0 : 1;
}

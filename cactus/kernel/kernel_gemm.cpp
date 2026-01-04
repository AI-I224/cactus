#include "kernel.h"
#include "kernel_utils.h"
#include <arm_neon.h>
#include <cstring>
#include <algorithm>
#include <cmath>

namespace {
    struct GemmBuffers {
        std::vector<int8_t> A_quant;
        std::vector<float> A_scales;
    };
    static thread_local GemmBuffers gemm_buffers;
}

static void cactus_matmul_f16_worker(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N,
    size_t start_row,
    size_t end_row
) {
    constexpr int TILE_M = 4;
    constexpr int TILE_N = 4;
    constexpr int VECTOR_WIDTH = 8;
    const size_t K_aligned = (K / (VECTOR_WIDTH * 2)) * (VECTOR_WIDTH * 2);

    for (size_t row_block = start_row; row_block < end_row; row_block += TILE_M) {
        for (size_t col_block = 0; col_block < N; col_block += TILE_N) {
            float16x8_t accumulators[TILE_M][TILE_N];
            for (int m = 0; m < TILE_M; ++m)
                for (int n = 0; n < TILE_N; ++n)
                    accumulators[m][n] = vdupq_n_f16(0.0);

            for (size_t k_block = 0; k_block < K_aligned; k_block += VECTOR_WIDTH * 2) {
                float16x8_t a_vec_low[TILE_M], a_vec_high[TILE_M];
                float16x8_t b_vec_low[TILE_N], b_vec_high[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        a_vec_low[m] = vld1q_f16(&a[row * K + k_block]);
                        a_vec_high[m] = vld1q_f16(&a[row * K + k_block + VECTOR_WIDTH]);
                    } else {
                        a_vec_low[m] = vdupq_n_f16(0.0);
                        a_vec_high[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        b_vec_low[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        b_vec_high[n] = vld1q_f16(&b_transposed[col * K + k_block + VECTOR_WIDTH]);
                    } else {
                        b_vec_low[n] = vdupq_n_f16(0.0);
                        b_vec_high[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n) {
                        accumulators[m][n] = accum_f16_dot(accumulators[m][n], 
                                                          a_vec_low[m], a_vec_high[m],
                                                          b_vec_low[n], b_vec_high[n]);
                    }
            }

            for (size_t k_block = K_aligned; k_block < K; k_block += VECTOR_WIDTH) {
                size_t remaining = K - k_block;
                float16x8_t a_vec[TILE_M], b_vec[TILE_N];

                for (int m = 0; m < TILE_M; ++m) {
                    size_t row = row_block + m;
                    if (row < M) {
                        if (remaining >= VECTOR_WIDTH) {
                            a_vec[m] = vld1q_f16(&a[row * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &a[row * K + k_block], remaining * sizeof(__fp16));
                            a_vec[m] = vld1q_f16(tmp);
                        }
                    } else {
                        a_vec[m] = vdupq_n_f16(0.0);
                    }
                }

                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col < N) {
                        if (remaining >= VECTOR_WIDTH) {
                            b_vec[n] = vld1q_f16(&b_transposed[col * K + k_block]);
                        } else {
                            __fp16 tmp[VECTOR_WIDTH] = {0.0};
                            memcpy(tmp, &b_transposed[col * K + k_block], remaining * sizeof(__fp16));
                            b_vec[n] = vld1q_f16(tmp);
                        }
                    } else {
                        b_vec[n] = vdupq_n_f16(0.0);
                    }
                }

                for (int m = 0; m < TILE_M; ++m)
                    for (int n = 0; n < TILE_N; ++n)
                        accumulators[m][n] = vfmaq_f16(accumulators[m][n], a_vec[m], b_vec[n]);
            }

            for (int m = 0; m < TILE_M; ++m) {
                size_t row = row_block + m;
                if (row >= M) continue;
                for (int n = 0; n < TILE_N; ++n) {
                    size_t col = col_block + n;
                    if (col >= N) continue;
                    float16x4_t low = vget_low_f16(accumulators[m][n]);
                    float16x4_t high = vget_high_f16(accumulators[m][n]);
                    float16x4_t sum_vec = vadd_f16(low, high);
                    __fp16 sum = vget_lane_f16(sum_vec, 0) + vget_lane_f16(sum_vec, 1) + 
                                vget_lane_f16(sum_vec, 2) + vget_lane_f16(sum_vec, 3);
                    c[row * N + col] = sum;
                }
            }
        }
    }
}

void cactus_matmul_f16(
    const __fp16* a,
    const __fp16* b_transposed,
    __fp16* c,
    size_t M,
    size_t K,
    size_t N
) {
    constexpr size_t TILE_M = 4;
    const size_t num_row_blocks = (M + TILE_M - 1) / TILE_M;

    CactusThreading::parallel_for(num_row_blocks, CactusThreading::Thresholds::SCALAR_EXPENSIVE,
        [=](size_t start_block, size_t end_block) {
            for (size_t block_idx = start_block; block_idx < end_block; ++block_idx) {
                size_t start_row = block_idx * TILE_M;
                size_t end_row = std::min(start_row + TILE_M, M);

                cactus_matmul_f16_worker(
                    a, b_transposed, c,
                    M, K, N,
                    start_row, end_row
                );
            }
        });
}

static inline float quantize_row_fp16_to_int8(const __fp16* src, int8_t* dst, size_t K) {
    float32x4_t max_vec0 = vdupq_n_f32(0.0f);
    float32x4_t max_vec1 = vdupq_n_f32(0.0f);
    size_t k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(v0))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(v0))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_low_f16(v1))));
        max_vec1 = vmaxq_f32(max_vec1, vabsq_f32(vcvt_f32_f16(vget_high_f16(v1))));
    }

    max_vec0 = vmaxq_f32(max_vec0, max_vec1);

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_low_f16(vals))));
        max_vec0 = vmaxq_f32(max_vec0, vabsq_f32(vcvt_f32_f16(vget_high_f16(vals))));
    }

    float max_abs = vmaxvq_f32(max_vec0);

    for (; k < K; k++) {
        float val = fabsf((float)src[k]);
        if (val > max_abs) max_abs = val;
    }

    float scale = max_abs / 127.0f;
    if (scale < 1e-10f) scale = 1e-10f;
    float inv_scale = 1.0f / scale;
    float32x4_t inv_scale_vec = vdupq_n_f32(inv_scale);

    k = 0;

    for (; k + 16 <= K; k += 16) {
        float16x8_t v0 = vld1q_f16(src + k);
        float16x8_t v1 = vld1q_f16(src + k + 8);

        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v0)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v0)), inv_scale_vec));
        int32x4_t i2 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(v1)), inv_scale_vec));
        int32x4_t i3 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(v1)), inv_scale_vec));

        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x4_t s2 = vqmovn_s32(i2);
        int16x4_t s3 = vqmovn_s32(i3);
        int16x8_t s01 = vcombine_s16(s0, s1);
        int16x8_t s23 = vcombine_s16(s2, s3);
        int8x8_t r0 = vqmovn_s16(s01);
        int8x8_t r1 = vqmovn_s16(s23);
        vst1q_s8(dst + k, vcombine_s8(r0, r1));
    }

    for (; k + 8 <= K; k += 8) {
        float16x8_t vals = vld1q_f16(src + k);
        int32x4_t i0 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_low_f16(vals)), inv_scale_vec));
        int32x4_t i1 = vcvtnq_s32_f32(vmulq_f32(vcvt_f32_f16(vget_high_f16(vals)), inv_scale_vec));
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int8x8_t result = vqmovn_s16(vcombine_s16(s0, s1));
        vst1_s8(dst + k, result);
    }

    for (; k < K; k++) {
        float val = (float)src[k] * inv_scale;
        int32_t q = (int32_t)roundf(val);
        q = std::max(-128, std::min(127, q));
        dst[k] = (int8_t)q;
    }

    return scale;
}

void cactus_matmul_int(
    const __fp16* A,
    const int8_t* B,
    const __fp16* B_scales,
    __fp16* C,
    size_t M, size_t K, size_t N,
    size_t group_size
) {
    if (M == 0 || K == 0 || N == 0) return;

    #if defined(__APPLE__) && defined(__arm64__)
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 8;
  #else 
      // Android, Raspberry Pi, Linux ARM, Graviton, etc.
      constexpr size_t TILE_M = 4;
      constexpr size_t TILE_N = 4;
  #endif

    const size_t num_groups = K / group_size;
    const size_t K_aligned = ((K + group_size - 1) / group_size) * group_size;

    const size_t quant_size = M * K_aligned;
    if (gemm_buffers.A_quant.size() < quant_size) {
        gemm_buffers.A_quant.resize(quant_size);
    }
    if (gemm_buffers.A_scales.size() < M) {
        gemm_buffers.A_scales.resize(M);
    }
    int8_t* A_quant = gemm_buffers.A_quant.data();
    float* A_scales = gemm_buffers.A_scales.data();

    CactusThreading::parallel_for(M, CactusThreading::Thresholds::ELEMENT_WISE,
        [&, A_quant, A_scales](size_t m_start, size_t m_end) {
            for (size_t m = m_start; m < m_end; m++) {
                A_scales[m] = quantize_row_fp16_to_int8(
                    A + m * K, A_quant + m * K_aligned, K);
            }
        });

    constexpr size_t MAX_GROUPS = 64;

    CactusThreading::parallel_for_2d_tiled(M, N, TILE_M, TILE_N,
        [&, A_quant, A_scales](size_t m_start, size_t m_end, size_t n_start, size_t n_end) {
            size_t actual_m = m_end - m_start;
            size_t actual_n = n_end - n_start;

            int32_t all_group_acc[MAX_GROUPS][TILE_M][TILE_N] = {{{0}}};

            for (size_t g = 0; g < num_groups; g++) {
                size_t k_base = g * group_size;

                for (size_t k_offset = 0; k_offset < group_size; k_offset += 64) {
                    int8x16_t b_vec0[TILE_N], b_vec1[TILE_N], b_vec2[TILE_N], b_vec3[TILE_N];
                    for (size_t ni = 0; ni < actual_n; ni++) {
                        const int8_t* b_ptr = B + (n_start + ni) * K + k_base + k_offset;
                        b_vec0[ni] = vld1q_s8(b_ptr);
                        b_vec1[ni] = vld1q_s8(b_ptr + 16);
                        b_vec2[ni] = vld1q_s8(b_ptr + 32);
                        b_vec3[ni] = vld1q_s8(b_ptr + 48);
                    }

                    for (size_t mi = 0; mi < actual_m; mi++) {
                        const int8_t* a_ptr = A_quant + (m_start + mi) * K_aligned + k_base + k_offset;
                        int8x16_t a_vec0 = vld1q_s8(a_ptr);
                        int8x16_t a_vec1 = vld1q_s8(a_ptr + 16);
                        int8x16_t a_vec2 = vld1q_s8(a_ptr + 32);
                        int8x16_t a_vec3 = vld1q_s8(a_ptr + 48);

                        for (size_t ni = 0; ni < actual_n; ni++) {
                            int32x4_t sum = vdupq_n_s32(0);
                            sum = accum_dot(sum, a_vec0, b_vec0[ni]);
                            sum = accum_dot(sum, a_vec1, b_vec1[ni]);
                            sum = accum_dot(sum, a_vec2, b_vec2[ni]);
                            sum = accum_dot(sum, a_vec3, b_vec3[ni]);
                            all_group_acc[g][mi][ni] += vaddvq_s32(sum);
                        }
                    }
                }
            }

            float acc[TILE_M][TILE_N] = {{0}};
            for (size_t mi = 0; mi < actual_m; mi++) {
                float a_scale = A_scales[m_start + mi];
                for (size_t ni = 0; ni < actual_n; ni++) {
                    float sum = 0.0f;
                    for (size_t g = 0; g < num_groups; g++) {
                        float b_scale = (float)B_scales[g * N + (n_start + ni)];
                        sum += (float)all_group_acc[g][mi][ni] * b_scale;
                    }
                    acc[mi][ni] = sum * a_scale;
                }
            }

            for (size_t mi = 0; mi < actual_m; mi++) {
                for (size_t ni = 0; ni < actual_n; ni++) {
                    C[(m_start + mi) * N + (n_start + ni)] = (__fp16)acc[mi][ni];
                }
            }
        });
}
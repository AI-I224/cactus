#include "graph.h"
#include "../kernel/kernel.h"
#include <cstring>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <assert.h>

namespace {
    thread_local std::vector<__fp16> transpose_buffer_fp16;

    void ensure_transpose_buffer_fp16(size_t required_size) {
        if (transpose_buffer_fp16.size() < required_size) {
            transpose_buffer_fp16.resize(required_size);
        }
    }
}

void shrink_thread_local_buffers() {
    std::vector<__fp16>().swap(transpose_buffer_fp16);
}

void compute_reduce_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    int axis = node.params.axis;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Reduction operations only support FP16 precision");
    }

    if (axis == -1) {
        switch (node.op_type) {
            case OpType::SUM: {
                double result = cactus_sum_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::MEAN: {
                double result = cactus_mean_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::VARIANCE: {
                double result = cactus_variance_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = static_cast<__fp16>(result);
                break;
            }
            case OpType::MIN: {
                __fp16 result = cactus_min_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = result;
                break;
            }
            case OpType::MAX: {
                __fp16 result = cactus_max_all_f16(input_buffer.data_as<__fp16>(), input_buffer.total_size);
                node.output_buffer.data_as<__fp16>()[0] = result;
                break;
            }
            default: break;
        }
    } else {
        const auto& shape = input_buffer.shape;
        size_t axis_idx = static_cast<size_t>(axis);

        size_t outer_size = 1;
        for (size_t i = 0; i < axis_idx; i++) {
            outer_size *= shape[i];
        }

        size_t axis_size = shape[axis_idx];

        size_t inner_size = 1;
        for (size_t i = axis_idx + 1; i < shape.size(); i++) {
            inner_size *= shape[i];
        }

        switch (node.op_type) {
            case OpType::SUM:
                cactus_sum_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    outer_size, axis_size, inner_size);
                break;
            case OpType::MEAN:
                cactus_mean_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    outer_size, axis_size, inner_size);
                break;
            case OpType::VARIANCE:
                cactus_variance_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                         outer_size, axis_size, inner_size);
                break;
            case OpType::MIN:
                cactus_min_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    outer_size, axis_size, inner_size);
                break;
            case OpType::MAX:
                cactus_max_axis_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                    outer_size, axis_size, inner_size);
                break;
            default: break;
        }
    }
}

void compute_fused_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    switch (node.op_type) {
        case OpType::GATHER: {
            const auto& tensor_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            size_t first_dim = tensor_buffer.shape[0];
            size_t element_size = 1;
            for (size_t i = 1; i < tensor_buffer.shape.size(); i++) {
                element_size *= tensor_buffer.shape[i];
            }
            
            size_t num_indices = indices_buffer.total_size;
            size_t bytes_per_element = element_size * PrecisionTraits::size_of(tensor_buffer.precision);
            
            if (tensor_buffer.precision == Precision::INT8) {
                const int8_t* tensor_data = tensor_buffer.data_as<int8_t>();
                int8_t* output = node.output_buffer.data_as<int8_t>();

                const bool is_grouped = tensor_buffer.is_grouped_int8();
                __fp16* gathered_scales = nullptr;
                const __fp16* src_scales = nullptr;
                size_t num_groups = 0;

                if (is_grouped) {
                    num_groups = tensor_buffer.num_groups;
                    src_scales = tensor_buffer.scales_as_fp16();
                    size_t scales_bytes = num_indices * num_groups * sizeof(__fp16);
                    node.output_buffer.owned_scales = std::make_unique<char[]>(scales_bytes);
                    gathered_scales = reinterpret_cast<__fp16*>(node.output_buffer.owned_scales.get());
                }

                if (indices_buffer.precision == Precision::INT8) {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                        if (is_grouped) {
                            for (size_t g = 0; g < num_groups; g++) {
                                gathered_scales[g * num_indices + i] = src_scales[g * first_dim + idx];
                            }
                        }
                    }
                } else {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                        if (is_grouped) {
                            for (size_t g = 0; g < num_groups; g++) {
                                gathered_scales[g * num_indices + i] = src_scales[g * first_dim + idx];
                            }
                        }
                    }
                }

                if (is_grouped) {
                    node.output_buffer.group_size = tensor_buffer.group_size;
                    node.output_buffer.num_groups = num_groups;
                    node.output_buffer.scales_data = gathered_scales;
                }
            } else if (tensor_buffer.precision == Precision::FP16) {
                const __fp16* tensor_data = tensor_buffer.data_as<__fp16>();
                __fp16* output = node.output_buffer.data_as<__fp16>();
                
                if (indices_buffer.precision == Precision::INT8) {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                } else {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                }
            } else {
                const float* tensor_data = tensor_buffer.data_as<float>();
                float* output = node.output_buffer.data_as<float>();
                
                if (indices_buffer.precision == Precision::INT8) {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                } else {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= first_dim) {
                            throw std::runtime_error("Gather index " + std::to_string(idx) + " out of bounds for dimension " + std::to_string(first_dim));
                        }
                        std::memcpy(output + i * element_size, tensor_data + idx * element_size, bytes_per_element);
                    }
                }
            }
            break;
        }
        case OpType::SLICE: {
            auto* input_node = nodes[node_index_map.at(node.input_ids[0])].get();
            auto& input_buffer = input_node->output_buffer;

            const size_t axis_index = static_cast<size_t>(node.params.axis);

            const size_t axis_size = input_buffer.shape[axis_index];
            const size_t slice_start = node.params.slice_start;
            size_t slice_length = node.params.slice_length;

            if (slice_length == 0) {
                slice_length = axis_size - slice_start;
            }

            const size_t element_size = PrecisionTraits::size_of(input_buffer.precision);

            if (axis_index == 0) {
                size_t inner_elements = 1;
                for (size_t i = 1; i < input_buffer.shape.size(); ++i) {
                    inner_elements *= input_buffer.shape[i];
                }

                auto* base_ptr = static_cast<char*>(input_buffer.get_data());
                if (!base_ptr) {
                    throw std::runtime_error("Slice input buffer is not available");
                }

                const size_t byte_offset = slice_start * inner_elements * element_size;

                node.output_buffer.set_external(base_ptr + byte_offset);
                node.output_buffer.precision = input_buffer.precision;

                if (input_buffer.is_grouped_int8()) {
                   size_t num_groups = input_buffer.num_groups;
                    size_t scales_bytes = slice_length * num_groups * sizeof(__fp16);
                    node.output_buffer.owned_scales = std::make_unique<char[]>(scales_bytes);
                    __fp16* sliced_scales = reinterpret_cast<__fp16*>(node.output_buffer.owned_scales.get());
                    const __fp16* input_scales = input_buffer.scales_as_fp16();

                    for (size_t g = 0; g < num_groups; g++) {
                        for (size_t i = 0; i < slice_length; i++) {
                            sliced_scales[g * slice_length + i] = input_scales[g * axis_size + (slice_start + i)];
                        }
                    }

                    node.output_buffer.group_size = input_buffer.group_size;
                    node.output_buffer.num_groups = num_groups;
                    node.output_buffer.scales_data = sliced_scales;
                }
                break;
            }

            const char* input_ptr = static_cast<const char*>(input_buffer.get_data());
            if (!input_ptr) {
                throw std::runtime_error("Slice input buffer is not available");
            }

            size_t inner_elements = 1;
            for (size_t i = axis_index + 1; i < input_buffer.shape.size(); ++i) {
                inner_elements *= input_buffer.shape[i];
            }

            size_t outer_elements = 1;
            for (size_t i = 0; i < axis_index; ++i) {
                outer_elements *= input_buffer.shape[i];
            }

            const size_t copy_block_elements = slice_length * inner_elements;
            const size_t axis_stride_elements = axis_size * inner_elements;
            const size_t copy_block_bytes = copy_block_elements * element_size;
            const size_t axis_stride_bytes = axis_stride_elements * element_size;

            node.output_buffer.external_data = nullptr;
            node.output_buffer.allocate();
            node.output_buffer.precision = input_buffer.precision;

            auto* output_ptr = static_cast<char*>(node.output_buffer.get_data());
            if (!output_ptr) {
                throw std::runtime_error("Slice output buffer could not be allocated");
            }

            for (size_t outer = 0; outer < outer_elements; ++outer) {
                const char* src = input_ptr + outer * axis_stride_bytes + slice_start * inner_elements * element_size;
                char* dst = output_ptr + outer * copy_block_bytes;
                std::memcpy(dst, src, copy_block_bytes);
            }
            break;
        }
        case OpType::EMBEDDING: {
            const auto& embeddings_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            
            size_t vocab_size = embeddings_buffer.shape[0];
            size_t hidden_dim = embeddings_buffer.shape[1];
            size_t num_indices = indices_buffer.total_size;
            
            if (embeddings_buffer.precision == Precision::INT8) {
                const int8_t* embeddings = embeddings_buffer.data_as<int8_t>();
                __fp16* output = node.output_buffer.data_as<__fp16>();

                if (embeddings_buffer.is_grouped_int8()) {
                    const __fp16* scales = embeddings_buffer.scales_as_fp16();
                    size_t group_size = embeddings_buffer.group_size;
                    size_t num_groups = embeddings_buffer.num_groups;

                    auto dequant_row = [&](size_t i, size_t idx) {
                        const int8_t* emb_row = embeddings + idx * hidden_dim;
                        __fp16* out_row = output + i * hidden_dim;

                        for (size_t g = 0; g < num_groups; g++) {
                            float scale = (float)scales[g * vocab_size + idx];
                            size_t k_start = g * group_size;
                            size_t k_end = std::min(k_start + group_size, hidden_dim);
                            for (size_t k = k_start; k < k_end; k++) {
                                out_row[k] = static_cast<__fp16>(emb_row[k] * scale);
                            }
                        }
                    };

                    if (indices_buffer.precision == Precision::FP32) {
                        const float* indices = indices_buffer.data_as<float>();
                        for (size_t i = 0; i < num_indices; i++) {
                            size_t idx = static_cast<size_t>(indices[i]);
                            if (idx >= vocab_size) {
                                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                            }
                            dequant_row(i, idx);
                        }
                    } else {
                        const int8_t* indices = indices_buffer.data_as<int8_t>();
                        for (size_t i = 0; i < num_indices; i++) {
                            size_t idx = static_cast<size_t>(indices[i]);
                            if (idx >= vocab_size) {
                                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                            }
                            dequant_row(i, idx);
                        }
                    }
                } else {
                    if (indices_buffer.precision == Precision::FP32) {
                        const float* indices = indices_buffer.data_as<float>();
                        for (size_t i = 0; i < num_indices; i++) {
                            size_t idx = static_cast<size_t>(indices[i]);
                            if (idx >= vocab_size) {
                                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                            }
                            for (size_t j = 0; j < hidden_dim; j++) {
                                output[i * hidden_dim + j] = static_cast<__fp16>(embeddings[idx * hidden_dim + j]);
                            }
                        }
                    } else {
                        const int8_t* indices = indices_buffer.data_as<int8_t>();
                        for (size_t i = 0; i < num_indices; i++) {
                            size_t idx = static_cast<size_t>(indices[i]);
                            if (idx >= vocab_size) {
                                throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                            }
                            for (size_t j = 0; j < hidden_dim; j++) {
                                output[i * hidden_dim + j] = static_cast<__fp16>(embeddings[idx * hidden_dim + j]);
                            }
                        }
                    }
                }
            } else if (embeddings_buffer.precision == Precision::FP16) {
                const __fp16* embeddings = embeddings_buffer.data_as<__fp16>();
                __fp16* output = node.output_buffer.data_as<__fp16>();

                if (indices_buffer.precision == Precision::FP32) {
                    const float* indices = indices_buffer.data_as<float>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                } else {
                    const int8_t* indices = indices_buffer.data_as<int8_t>();
                    for (size_t i = 0; i < num_indices; i++) {
                        size_t idx = static_cast<size_t>(indices[i]);
                        if (idx >= vocab_size) {
                            throw std::runtime_error("Embedding index out of bounds: " + std::to_string(idx) + " >= " + std::to_string(vocab_size));
                        }
                        for (size_t j = 0; j < hidden_dim; j++) {
                            output[i * hidden_dim + j] = embeddings[idx * hidden_dim + j];
                        }
                    }
                }
            } else {
                throw std::runtime_error("Embedding only supports INT8 and FP16 precision");
            }
            break;
        }
        case OpType::BILINEAR_INTERPOLATION: {
            const auto& pos_embeds_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

            size_t total_pos_embeds = pos_embeds_buffer.shape[0];
            size_t embed_dim = pos_embeds_buffer.shape[1];

            size_t src_height = static_cast<size_t>(std::sqrt(total_pos_embeds));
            size_t src_width = src_height;

            size_t dst_height = node.params.dst_height;
            size_t dst_width = node.params.dst_width;

            __fp16* output = node.output_buffer.data_as<__fp16>();

            if (pos_embeds_buffer.precision == Precision::FP16) {
                const __fp16* input = pos_embeds_buffer.data_as<__fp16>();
                cactus_bilinear_interpolation_f16(input, output, src_height, src_width, embed_dim,
                                                  dst_height, dst_width);
            }
            else if (pos_embeds_buffer.precision == Precision::INT8) {
                std::vector<__fp16> input_fp16(total_pos_embeds * embed_dim);
                cactus_int8_to_fp16(pos_embeds_buffer.data_as<int8_t>(), input_fp16.data(),
                                    total_pos_embeds * embed_dim);
                cactus_bilinear_interpolation_f16(input_fp16.data(), output, src_height, src_width, embed_dim,
                                                  dst_height, dst_width);
            }
            else {
                throw std::runtime_error("BILINEAR_INTERPOLATION only supports INT8 and FP16 input precision");
            }
            break;
        }
        case OpType::RMS_NORM: {
            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

            if (input_buffer.shape.size() != 2) {
                throw std::runtime_error("RMS normalization requires 2D input tensor [batch_size, dims], got " +
                                        std::to_string(input_buffer.shape.size()) + "D tensor");
            }

            size_t batch_size = input_buffer.shape[0];
            size_t dims = input_buffer.shape[1];

            if (input_buffer.precision != Precision::FP16) {
                throw std::runtime_error("RMS normalization only supports FP16 precision");
            }

            cactus_rms_norm_f16(input_buffer.data_as<__fp16>(), weight_buffer.data_as<__fp16>(),
               node.output_buffer.data_as<__fp16>(), batch_size, dims, node.params.epsilon);
            break;
        }
        case OpType::ROPE: {
            if (node.params.backend == ComputeBackend::NPU) {
                throw std::runtime_error("NPU RoPE operation not yet implemented");
            }

            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& shape = input_buffer.shape;

            if (shape.size() < 4) {
                throw std::runtime_error("RoPE operation requires 4D tensor with shape [batch, seq_len, num_heads, head_dim], got " +
                                        std::to_string(shape.size()) + "D tensor");
            }

            if (input_buffer.precision != Precision::FP16 || node.output_buffer.precision != Precision::FP16) {
                throw std::runtime_error("RoPE operation only supports FP16 precision");
            }

            size_t batch_size = shape[0];
            size_t seq_len = shape[1];
            size_t num_heads = shape[2];
            size_t head_dim = shape[3];

            cactus_rope_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                           batch_size, seq_len, num_heads, head_dim, node.params.position_offset, node.params.theta);
            break;
        }
        case OpType::SOFTMAX: {
            const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& shape = input_buffer.shape;

            if (shape.size() < 2) {
                throw std::runtime_error("Softmax operation requires at least 2D tensor, got " +
                                        std::to_string(shape.size()) + "D tensor");
            }

            if (input_buffer.precision != Precision::FP16) {
                throw std::runtime_error("Softmax operation only supports FP16 precision");
            }

            size_t batch_size = 1;
            for (size_t i = 0; i < shape.size() - 1; i++) {
                batch_size *= shape[i];
            }
            size_t vocab_size = shape[shape.size() - 1];

            cactus_softmax_f16(input_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                              batch_size, 1, vocab_size);
            break;
        }
        case OpType::ATTENTION: {
            if (node.params.backend == ComputeBackend::NPU) {
                throw std::runtime_error("NPU attention operation not yet implemented");
            }

            if (node.input_ids.size() < 3) {
                throw std::runtime_error("Attention operation requires 3 inputs (query, key, value), got " +
                                        std::to_string(node.input_ids.size()) + " inputs");
            }

            const auto& query_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& key_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            const auto& value_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
            const auto& q_shape = query_buffer.shape;
            const auto& k_shape = key_buffer.shape;

            if (q_shape.size() < 4) {
                throw std::runtime_error("Attention operation requires 4D tensors [batch, seq_len, num_heads, head_dim], got " +
                                        std::to_string(q_shape.size()) + "D tensor");
            }

            if (query_buffer.precision != Precision::FP16) {
                throw std::runtime_error("Attention operation only supports FP16 precision");
            }

            size_t batch_size = q_shape[0];
            size_t seq_len = q_shape[1];
            size_t num_q_heads = q_shape[2];
            size_t head_dim = q_shape[3];
            size_t num_kv_heads = k_shape[2];
            size_t kv_seq_len = key_buffer.shape[1];

            cactus_attention_f16(query_buffer.data_as<__fp16>(), key_buffer.data_as<__fp16>(),
                                 value_buffer.data_as<__fp16>(), node.output_buffer.data_as<__fp16>(),
                                 batch_size, seq_len, kv_seq_len, num_q_heads, num_kv_heads, head_dim, node.params.scale, nullptr,
                                 node.params.position_offset, node.params.window_size, node.params.is_causal);
            break;
        }
        case OpType::ATTENTION_INT8_HYBRID: {
            const auto& query_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& key_new_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
            const auto& value_new_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
            const auto& q_shape = query_buffer.shape;

            if (q_shape.size() < 4) {
                throw std::runtime_error("ATTENTION_INT8_HYBRID requires 4D query tensor");
            }

            size_t batch_size = q_shape[0];
            size_t seq_len = q_shape[1];
            size_t num_q_heads = q_shape[2];
            size_t head_dim = node.params.head_dim;
            size_t num_kv_heads = node.params.num_kv_heads;
            size_t cache_len = node.params.cache_seq_len;
            size_t new_len = key_new_buffer.shape[1];

            cactus_attention_hybrid_int8_fp16(
                query_buffer.data_as<__fp16>(),
                node.params.cached_keys_int8,
                node.params.cached_values_int8,
                node.params.cached_k_scales,
                node.params.cached_v_scales,
                key_new_buffer.data_as<__fp16>(),
                value_new_buffer.data_as<__fp16>(),
                node.output_buffer.data_as<__fp16>(),
                batch_size, seq_len, cache_len, new_len,
                num_q_heads, num_kv_heads, head_dim,
                node.params.scale, node.params.position_offset, true
            );
            break;
        }
        case OpType::CONV1D_CAUSAL: {
            if (node.params.backend == ComputeBackend::NPU) {
                throw std::runtime_error("NPU causal convolution operation not yet implemented");
            }

            const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer; 
            const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer; 
            auto& Y = node.output_buffer;

            if (X.shape.size() != 3) {
                throw std::runtime_error("Causal conv requires 3D input [batch, seq_len, in_channels]");
            }
            if (W.shape.size() != 3) {
                throw std::runtime_error("Weight must be 3D");
            }

            const size_t N     = X.shape[0];
            const size_t L     = X.shape[1];
            const size_t C_in  = X.shape[2];
            const size_t W0    = W.shape[0];
            const size_t W1    = W.shape[1]; 
            const size_t K     = W.shape[2];
            const size_t dil   = node.params.dilation; 
            if (dil < 1) throw std::runtime_error("dilation must be >= 1");

            size_t M = 1;
            size_t C_out = 0;

            assert((W1 == 1) && (W0 % C_in == 0) && "Only depthwise causal convolution is supported currently");
            M = W0 / C_in;
            C_out = C_in * M;
            
            Y.shape = { N, L, C_out };
            Y.precision = X.precision;
            
            if (W.precision == Precision::INT8) {
                const size_t W_size = W0 * W1 * K;
                const int8_t* W_int8 = W.data_as<int8_t>();

                std::vector<__fp16> W_fp16(W_size);

                if (W.is_grouped_int8()) {
                    const __fp16* scales = W.scales_as_fp16();
                    const size_t K_total = W1 * K;
                    const size_t group_size = W.group_size;
                    const size_t num_groups = W.num_groups;

                    for (size_t row = 0; row < W0; ++row) {
                        for (size_t col = 0; col < K_total; ++col) {
                            size_t idx = row * K_total + col;
                            size_t group_idx = col / group_size;
                            float scale = static_cast<float>(scales[group_idx * W0 + row]);
                            W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                        }
                    }
                } else {
                    for (size_t i = 0; i < W_size; ++i) {
                        W_fp16[i] = static_cast<__fp16>(W_int8[i]);
                    }
                }

                cactus_conv1d_causal_depthwise_f16(
                    X.data_as<__fp16>(), W_fp16.data(), Y.data_as<__fp16>(),
                    N, L, C_in, K, dil);
            } else if (W.precision == Precision::FP16) {
                cactus_conv1d_causal_depthwise_f16(
                    X.data_as<__fp16>(), W.data_as<__fp16>(), Y.data_as<__fp16>(),
                    N, L, C_in, K, dil);
            } else {
                throw std::runtime_error("Depthwise causal conv supports INT8/FP16 weights");
            }
            break;
        }

        case OpType::CONV1D_K3: {

                if (node.params.backend == ComputeBackend::NPU) {
                    throw std::runtime_error("NPU causal convolution operation not yet implemented");
                }

                const auto& X = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
                const auto& W = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
                auto& Y = node.output_buffer;

                if (X.shape.size() != 3)
                    throw std::runtime_error("Conv requires 3D input [N, C_in, L]!");

                if (W.shape.size() != 3)
                    throw std::runtime_error("Weight must be [C_out, C_in, 3]!");

                const size_t N    = X.shape[0];
                const size_t C_in = X.shape[1];
                const size_t L    = X.shape[2];

                const size_t C_out = W.shape[0];
                const size_t K     = W.shape[2];
                const size_t stride = node.params.stride;

                if (K != 3)
                    throw std::runtime_error("Conv1d_k3 only supports K=3!");

                size_t L_out = ((L - 1) / stride) + 1;
                Y.shape     = { N, C_out, L_out };
                Y.precision = X.precision;

                if (X.precision != Precision::FP16) {
                    throw std::runtime_error("Conv1d_k3 only supports FP16 activations");
                }

                if (W.precision == Precision::INT8) {
                    const size_t W_size = C_out * C_in * K;
                    const int8_t* W_int8 = W.data_as<int8_t>();

                    std::vector<__fp16> W_fp16(W_size);

                    if (W.is_grouped_int8()) {
                        const __fp16* scales = W.scales_as_fp16();
                        const size_t K_total = C_in * K;
                        const size_t group_size = W.group_size;
                        const size_t num_groups = W.num_groups;

                        for (size_t row = 0; row < C_out; ++row) {
                            for (size_t col = 0; col < K_total; ++col) {
                                size_t idx = row * K_total + col;
                                size_t group_idx = col / group_size;
                                float scale = static_cast<float>(scales[group_idx * C_out + row]);
                                W_fp16[idx] = static_cast<__fp16>(W_int8[idx] * scale);
                            }
                        }
                    } else {
                        for (size_t i = 0; i < W_size; ++i) {
                            W_fp16[i] = static_cast<__fp16>(W_int8[i]);
                        }
                    }

                    cactus_conv1d_f16_k3(
                        X.data_as<__fp16>(),
                        W_fp16.data(),
                        Y.data_as<__fp16>(),
                        N, L, C_in, C_out, stride
                    );
                } else if (W.precision == Precision::FP16) {
                    cactus_conv1d_f16_k3(
                        X.data_as<__fp16>(),
                        W.data_as<__fp16>(),
                        Y.data_as<__fp16>(),
                        N, L, C_in, C_out, stride
                    );
                } else {
                    throw std::runtime_error("Conv1d_k3 only supports FP16 and INT8 weights");
                }

                break;
        }


        case OpType::CONCAT: {
            const auto& input1_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
            const auto& input2_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

            std::vector<size_t> shape1 = input1_buffer.shape;
            std::vector<size_t> shape2 = input2_buffer.shape;
            std::vector<size_t> output_shape = node.output_buffer.shape;

            if (input1_buffer.precision != Precision::FP16) {
                throw std::runtime_error("Concat operation only supports FP16 precision");
            }
            cactus_concat_f16(input1_buffer.data_as<__fp16>(), input2_buffer.data_as<__fp16>(),
                             node.output_buffer.data_as<__fp16>(),
                             shape1.data(), shape2.data(), output_shape.data(),
                             shape1.size(), node.params.axis);
            break;
        }
        default: break;
    }
}

void compute_transpose_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    if (node.params.backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU transpose operation not yet implemented");
    }

    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

    if (input_buffer.precision != Precision::FP16) {
        throw std::runtime_error("Transpose only supports FP16 precision");
    }

    const auto& permutation = node.params.permutation;

    const __fp16* input = input_buffer.data_as<__fp16>();
    __fp16* output = node.output_buffer.data_as<__fp16>();
    cactus_transpose_f16(input, output, input_buffer.shape.data(), permutation.data(), permutation.size(), 0, input_buffer.total_size);
}


void compute_matmul_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& lhs_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& rhs_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& lhs_shape = lhs_buffer.shape;
    const auto& rhs_shape = rhs_buffer.shape;
    
    size_t M = lhs_shape[lhs_shape.size() - 2];
    size_t K = lhs_shape[lhs_shape.size() - 1];
    size_t N = node.params.pretransposed_rhs ? 
               rhs_shape[rhs_shape.size() - 2] : rhs_shape[rhs_shape.size() - 1];
    
    bool pretransposed_rhs = node.params.pretransposed_rhs;
    
    ComputeBackend backend = node.params.backend;
    
    if (backend == ComputeBackend::NPU) {
        throw std::runtime_error("NPU matrix multiplication not yet implemented");
    }
    
    if (lhs_buffer.precision == Precision::FP16 && rhs_buffer.is_grouped_int8()) {
        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        const int8_t* rhs = rhs_buffer.data_as<int8_t>();
        const __fp16* rhs_scales = rhs_buffer.scales_as_fp16();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (!pretransposed_rhs) {
            throw std::runtime_error("Group-wise INT8 matmul requires pretransposed weights");
        }

        cactus_matmul_int(lhs, rhs, rhs_scales, output,
                                   M, K, N, rhs_buffer.group_size);

    } else {
        if (lhs_buffer.precision != Precision::FP16) {
            throw std::runtime_error("Matmul only supports FP16 precision for activations");
        }

        const __fp16* lhs = lhs_buffer.data_as<__fp16>();
        const __fp16* rhs = rhs_buffer.data_as<__fp16>();
        __fp16* output = node.output_buffer.data_as<__fp16>();

        if (pretransposed_rhs) {
            cactus_matmul_f16(lhs, rhs, output, M, K, N);
        } else {
            size_t transpose_size = rhs_shape[0] * rhs_shape[1];
            ensure_transpose_buffer_fp16(transpose_size);

            cactus_transpose_2d_f16(rhs, transpose_buffer_fp16.data(),
                                    rhs_shape[0], rhs_shape[1], 0, rhs_shape[0]);
            cactus_matmul_f16(lhs, transpose_buffer_fp16.data(), output, M, K, N);
        }
    }
}

void compute_sample_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& logits_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;

    float temperature = node.params.temperature;
    float top_p = node.params.top_p;
    size_t top_k = node.params.top_k;
    size_t random_seed = node.params.random_seed;

    const float* bias_values = node.params.bias_values.empty() ? nullptr : node.params.bias_values.data();
    const uint32_t* bias_indices = node.params.bias_indices.empty() ? nullptr : node.params.bias_indices.data();
    size_t bias_count = node.params.bias_values.size();

    if (logits_buffer.shape.size() != 2) {
        throw std::runtime_error("Sample expects 2D logits tensor [seq_len, vocab_size]");
    }

    size_t seq_len = logits_buffer.shape[0];
    size_t vocab_size = logits_buffer.shape[1];
    size_t last_token_offset = (seq_len - 1) * vocab_size;

    if (logits_buffer.precision == Precision::FP16) {
        const __fp16* logits_fp16 = logits_buffer.data_as<__fp16>();
        cactus_sample_f16(logits_fp16 + last_token_offset, node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count);
    } else {
        const float* logits_fp32 = logits_buffer.data_as<float>();
        cactus_sample_f32(logits_fp32 + last_token_offset, node.output_buffer.data_as<uint32_t>(),
                         vocab_size, temperature, top_p, top_k, random_seed,
                         bias_values, bias_indices, bias_count);
    }
}

void compute_scatter_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& indices_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& values_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;

    if (indices_buffer.shape != values_buffer.shape) {
        throw std::runtime_error("ScatterTopK requires indices and values with identical shapes");
    }
    if (indices_buffer.shape.size() != 2) {
        throw std::runtime_error("ScatterTopK currently supports 2D tensors");
    }

    size_t batch_size = indices_buffer.shape[0];
    size_t top_k = indices_buffer.shape[1];
    size_t num_classes = node.params.num_classes;

    if (num_classes == 0) {
        throw std::runtime_error("ScatterTopK requires num_classes > 0");
    }

    float* output = node.output_buffer.data_as<float>();
    std::fill(output, output + num_classes * batch_size, 0.0f);

    if (indices_buffer.precision != Precision::FP32 || values_buffer.precision != Precision::FP32) {
        throw std::runtime_error("ScatterTopK currently expects FP32 inputs");
    }

    const float* indices_data = indices_buffer.data_as<float>();
    const float* values_data = values_buffer.data_as<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t k = 0; k < top_k; ++k) {
            float raw_index = indices_data[b * top_k + k];
            if (!std::isfinite(raw_index)) {
                throw std::runtime_error("ScatterTopK index is not finite");
            }
            size_t expert_index = static_cast<size_t>(raw_index + 0.5f);
            if (expert_index >= num_classes) {
                throw std::runtime_error("ScatterTopK index out of range");
            }
            float weight = values_data[b * top_k + k];
            output[expert_index * batch_size + b] = weight;
        }
    }
}

void compute_topk_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;    
    if (input_buffer.shape.size() != 2) {
        throw std::runtime_error("TopK currently only supports 2D tensors [batch, features]");
    }
    
    size_t k = node.params.top_k;
    size_t batch_size = input_buffer.shape[0];
    size_t feature_size = input_buffer.shape[1];
    size_t block_size = batch_size * k;
    
    std::vector<float> input_float(input_buffer.total_size);
    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("TopK currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        const float* input_fp32 = input_buffer.data_as<float>();
        std::memcpy(input_float.data(), input_fp32, input_buffer.total_size * sizeof(float));
    }
    
    float* output = node.output_buffer.data_as<float>();
    
    for (size_t b = 0; b < batch_size; ++b) {
        const float* row = input_float.data() + b * feature_size;
        
        std::vector<std::pair<size_t, float>> indexed_values(feature_size);
        for (size_t i = 0; i < feature_size; ++i) {
            indexed_values[i] = {i, row[i]};
        }
        
        std::partial_sort(indexed_values.begin(), 
                         indexed_values.begin() + k, 
                         indexed_values.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        float* idx_out_row = output + b * k;
        float* val_out_row = output + block_size + b * k;
        for (size_t i = 0; i < k; ++i) {
            idx_out_row[i] = static_cast<float>(indexed_values[i].first);
            val_out_row[i] = indexed_values[i].second;
        }
    }
}

void compute_layernorm_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& weight_buffer = nodes[node_index_map.at(node.input_ids[1])]->output_buffer;
    const auto& bias_buffer = nodes[node_index_map.at(node.input_ids[2])]->output_buffer;
    float epsilon = node.params.epsilon;
    
    if (input_buffer.shape.empty()) {
        throw std::runtime_error("LayerNorm requires non-empty input tensor");
    }
    
    size_t feature_size = input_buffer.shape.back();
    size_t batch_size = input_buffer.total_size / feature_size;
    
    std::vector<float> input_float(input_buffer.total_size);
    std::vector<float> weight_float(feature_size);
    std::vector<float> bias_float(feature_size);
    
    if (input_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 input");
    } else if (input_buffer.precision == Precision::FP16) {
        const __fp16* input_fp16 = input_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            input_float[i] = static_cast<float>(input_fp16[i]);
        }
    } else {
        std::memcpy(input_float.data(), input_buffer.data_as<float>(), input_buffer.total_size * sizeof(float));
    }
    
    if (weight_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 weight");
    } else if (weight_buffer.precision == Precision::FP16) {
        const __fp16* weight_fp16 = weight_buffer.data_as<__fp16>();
        for (size_t i = 0; i < feature_size; ++i) {
            weight_float[i] = static_cast<float>(weight_fp16[i]);
        }
    } else {
        std::memcpy(weight_float.data(), weight_buffer.data_as<float>(), feature_size * sizeof(float));
    }
    
    if (bias_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 bias");
    } else if (bias_buffer.precision == Precision::FP16) {
        const __fp16* bias_fp16 = bias_buffer.data_as<__fp16>();
        for (size_t i = 0; i < feature_size; ++i) {
            bias_float[i] = static_cast<float>(bias_fp16[i]);
        }
    } else {
        std::memcpy(bias_float.data(), bias_buffer.data_as<float>(), feature_size * sizeof(float));
    }
    
    std::vector<float> output_float(input_buffer.total_size);
    for (size_t b = 0; b < batch_size; ++b) {
        const float* input_row = input_float.data() + b * feature_size;
        float* output_row = output_float.data() + b * feature_size;
        
        float mean = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            mean += input_row[i];
        }
        mean /= feature_size;
        
        float variance = 0.0f;
        for (size_t i = 0; i < feature_size; ++i) {
            float diff = input_row[i] - mean;
            variance += diff * diff;
        }
        variance /= feature_size;
        
        float std_inv = 1.0f / std::sqrt(variance + epsilon);
        for (size_t i = 0; i < feature_size; ++i) {
            output_row[i] = (input_row[i] - mean) * std_inv * weight_float[i] + bias_float[i];
        }
    }
    
    if (node.output_buffer.precision == Precision::INT8) {
        throw std::runtime_error("LayerNorm currently does not support INT8 output");
    } else if (node.output_buffer.precision == Precision::FP16) {
        __fp16* output_fp16 = node.output_buffer.data_as<__fp16>();
        for (size_t i = 0; i < input_buffer.total_size; ++i) {
            output_fp16[i] = static_cast<__fp16>(output_float[i]);
        }
    } else {
        std::memcpy(node.output_buffer.data_as<float>(), output_float.data(), input_buffer.total_size * sizeof(float));
    }
}

void compute_index_node(GraphNode& node, const std::vector<std::unique_ptr<GraphNode>>& nodes, const std::unordered_map<size_t, size_t>& node_index_map) {
    const auto& input_buffer = nodes[node_index_map.at(node.input_ids[0])]->output_buffer;
    const auto& input_shape = input_buffer.shape;
    
    int dim = node.params.axis;
    size_t index_value = node.params.index_value;

    size_t element_size = PrecisionTraits::size_of(input_buffer.precision);
    const char* input_data = static_cast<const char*>(input_buffer.get_data());
    char* output_data = static_cast<char*>(node.output_buffer.get_data());

    if (dim == 0) {
        size_t slice_size = input_buffer.total_size / input_shape[0];
        size_t offset_bytes = index_value * slice_size * element_size;
        node.output_buffer.set_external(const_cast<char*>(input_data) + offset_bytes);
        return;
    }
    
    std::vector<size_t> input_strides(input_shape.size());
    input_strides[input_shape.size() - 1] = 1;
    for (int i = static_cast<int>(input_shape.size()) - 2; i >= 0; --i) {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }
    
    size_t slice_size = input_strides[dim];
    size_t outer_size = input_buffer.total_size / input_strides[dim - 1];
    size_t dim_stride = input_strides[dim];
    size_t block_size = dim_stride * input_shape[dim];
    
    size_t output_idx = 0;
    for (size_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
        size_t input_base = outer_idx * block_size + index_value * dim_stride;
        
        std::memcpy(output_data + output_idx * element_size,
                    input_data + input_base * element_size,
                    slice_size * element_size);
        
        output_idx += slice_size;
    }
}

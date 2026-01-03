import numpy as np
import struct
from pathlib import Path
from typing import Optional, Dict, Any, List

try:
    import torch
except ImportError:
    torch = None


GROUP_SIZE = 128 


def save_tensor_with_header(tensor, output_path, precision='FP16', transpose=False, stats_tracker=None, args=None, model_type=None):
    """Save a tensor to binary format with header metadata and group-wise INT8 quantization."""
    if torch is not None and isinstance(tensor, torch.Tensor):
        data = tensor.detach().cpu().numpy()
    else:
        data = np.array(tensor)

    original_data = data.copy()

    if model_type == 'gemma' and 'norm' in str(output_path):
        data = data + 1.0
        original_data = data.copy()

    if precision == 'INT8':
        filename = output_path.name
        if any(x in filename for x in ['norm', 'bias', 'vision']) or (model_type == 'bert' and 'embedding' in filename):
            precision = 'FP16'

    shape = list(data.shape)
    if transpose and len(shape) == 2:
        data = data.T
        original_data = original_data.T
        shape = [shape[1], shape[0]]

    if precision == 'INT8':
        if len(shape) == 2:
            N, K = shape

            if K % GROUP_SIZE != 0:
                pad_k = GROUP_SIZE - (K % GROUP_SIZE)
                data = np.pad(data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                original_data = np.pad(original_data, ((0, 0), (0, pad_k)), mode='constant', constant_values=0)
                K = data.shape[1]
                shape = [N, K]

            num_groups = K // GROUP_SIZE

            data_grouped = data.reshape(N, num_groups, GROUP_SIZE)
            original_grouped = original_data.reshape(N, num_groups, GROUP_SIZE)

            group_abs_max = np.max(np.abs(data_grouped), axis=2) 
            scales = (group_abs_max / 127.0).astype(np.float32)
            scales = np.maximum(scales, 1e-10)  

            quantized = np.clip(
                np.round(data_grouped / scales[:, :, np.newaxis]),
                -128, 127
            ).astype(np.int8)
            quantized_flat = quantized.reshape(N, K)

            dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(N, K)
            mse_error = np.mean((original_data - dequantized) ** 2)
            snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')

            original_flat = original_data.flatten()
            dequant_flat = dequantized.flatten()
            cos_sim = np.dot(original_flat, dequant_flat) / (np.linalg.norm(original_flat) * np.linalg.norm(dequant_flat) + 1e-10)

            scales_fp16 = scales.T.astype(np.float16)

        elif len(shape) == 1:
            K = shape[0]

            if K % GROUP_SIZE != 0:
                pad_k = GROUP_SIZE - (K % GROUP_SIZE)
                data = np.pad(data, (0, pad_k), mode='constant', constant_values=0)
                original_data = np.pad(original_data, (0, pad_k), mode='constant', constant_values=0)
                K = data.shape[0]
                shape = [K]

            num_groups = K // GROUP_SIZE
            N = 1

            data_grouped = data.reshape(1, num_groups, GROUP_SIZE)
            original_grouped = original_data.reshape(1, num_groups, GROUP_SIZE)

            group_abs_max = np.max(np.abs(data_grouped), axis=2)
            scales = (group_abs_max / 127.0).astype(np.float32)
            scales = np.maximum(scales, 1e-10)

            quantized = np.clip(
                np.round(data_grouped / scales[:, :, np.newaxis]),
                -128, 127
            ).astype(np.int8)
            quantized_flat = quantized.reshape(K)

            dequantized = (quantized.astype(np.float32) * scales[:, :, np.newaxis]).reshape(K)
            mse_error = np.mean((original_data - dequantized) ** 2)
            snr_db = 10 * np.log10(np.var(original_data) / mse_error) if mse_error > 0 else float('inf')
            cos_sim = np.dot(original_data, dequantized) / (np.linalg.norm(original_data) * np.linalg.norm(dequantized) + 1e-10)

            scales_fp16 = scales.T.astype(np.float16)
        else:
            precision = 'FP16'

    if precision == 'INT8':
        if stats_tracker:
            stats_tracker['quantized_tensors'] += 1
            stats_tracker['quantized_parameters'] += original_data.size
            stats_tracker['mse_values'].append(mse_error)
            stats_tracker['snr_values'].append(snr_db)
            stats_tracker['cos_sim_values'].append(cos_sim)

        with open(output_path, 'wb') as f:
            ndim = len(shape)
            f.write(struct.pack('<I', ndim))
            for dim in shape:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', 0))
            byte_size = quantized_flat.size
            f.write(struct.pack('<Q', byte_size))
            f.write(struct.pack('<I', GROUP_SIZE))
            f.write(struct.pack('<Q', num_groups))
            f.write(quantized_flat.tobytes())
            f.write(scales_fp16.tobytes())

        if stats_tracker:
            stats_tracker['total_tensors'] += 1
            stats_tracker['total_parameters'] += original_data.size

        return

    data = data.astype(np.float16)

    if stats_tracker:
        stats_tracker['total_tensors'] += 1
        stats_tracker['total_parameters'] += original_data.size

    data_flat = data.flatten()

    with open(output_path, 'wb') as f:
        ndim = len(shape)
        f.write(struct.pack('<I', ndim))
        for dim in shape:
            f.write(struct.pack('<Q', dim))

        f.write(struct.pack('<I', 1))  # FP16 precision

        byte_size = data_flat.size * 2  # FP16 = 2 bytes
        f.write(struct.pack('<Q', byte_size))

        f.write(data_flat.tobytes())


def format_config_value(value):
    """Format a config value for writing to config.txt."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return ','.join(str(v) for v in value)
    return str(value)


def create_quantization_stats():
    """Create an empty stats tracker dictionary for quantization metrics."""
    return {
        'total_tensors': 0,
        'quantized_tensors': 0,
        'total_parameters': 0,
        'quantized_parameters': 0,
        'mse_values': [],
        'snr_values': [],
        'cos_sim_values': [],
        'saturation_warnings': 0
    }


def print_quantization_summary(quantization_stats, args=None):
    """Print a summary of quantization statistics."""
    if quantization_stats['quantized_tensors'] > 0:
        mse_values = np.array(quantization_stats['mse_values'])
        snr_values = np.array(quantization_stats['snr_values'])
        cos_sim_values = np.array(quantization_stats['cos_sim_values'])

        print("\nQuantization Summary:")
        print(f"MSE - Mean: {np.mean(mse_values):.2e}, Max: {np.max(mse_values):.2e}, Median: {np.median(mse_values):.2e}, Min: {np.min(mse_values):.2e}")
        print(f"SNR - Mean: {np.mean(snr_values):.1f}dB, Max: {np.max(snr_values):.1f}dB, Median: {np.median(snr_values):.1f}dB, Min: {np.min(snr_values):.1f}dB")
        print(f"CosSim - Mean: {np.mean(cos_sim_values):.6f}, Max: {np.mean(cos_sim_values):.6f}, Median: {np.median(cos_sim_values):.6f}, Min: {np.min(cos_sim_values):.6f}")
        fp16_tensors = quantization_stats['total_tensors'] - quantization_stats['quantized_tensors']
        low_snr_fallbacks = quantization_stats.get('low_snr_fallbacks', 0)
        snr_threshold = getattr(args, 'snr_threshold', 30.0) if args else 30.0
        print(f"Processed {quantization_stats['quantized_tensors']} INT8 tensors, {fp16_tensors} FP16 tensors ({low_snr_fallbacks} SNR<{snr_threshold}dB fallbacks)")

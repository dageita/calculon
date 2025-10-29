"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""
Offline profiler for operator-level performance measurement in Calculon.
This module provides offline profiling capabilities similar to Crius but integrated with Calculon's architecture.
"""

import os
import pickle
import csv
import json
import time
import gc
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    from torch import nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Offline profiling will be limited.")

from calculon import *


@dataclass
class OfflineProfileConfigs:
    """Configuration for offline profiling in Calculon."""
    
    # Profiling ranges
    min_batch_size: int = 1
    max_batch_size: int = 32
    batch_size_step: Union[int, str] = "power-of-2"
    
    min_seq_len: int = 1
    max_seq_len: int = 1024
    seq_len_step: Union[int, str] = "power-of-2"
    
    # GEMM profiling dimensions
    gemm_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    
    # Data type and device
    dtype: str = "float16"  # "float16", "float32", "bfloat16"
    device: str = "cuda:0"
    
    # Profiling parameters
    num_warmup_steps: int = 5
    num_profile_steps: int = 10
    
    # Storage
    data_dir: str = "./calculon_offline_data"
    data_filename: str = "operator_profiles.pkl"
    csv_filename: str = "operator_profiles.csv"
    
    # Force overwrite existing data
    force_overwrite: bool = False
    
    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)


class CalculonOfflineProfiler:
    """Offline profiler for operator-level performance measurement in Calculon."""
    
    def __init__(self, configs: OfflineProfileConfigs, system: System = None):
        self.configs = configs
        self.system = system
        self.data_path = os.path.join(configs.data_dir, configs.data_filename)
        self.csv_path = os.path.join(configs.data_dir, configs.csv_filename)
        
        # Profiled data storage
        self.profiled_data = {}
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing profiled data from storage."""
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    self.profiled_data = pickle.load(f)
                print(f"Loaded {len(self.profiled_data)} profiled operator entries")
            except Exception as e:
                print(f"Failed to load existing data: {e}")
                self.profiled_data = {}
        else:
            self.profiled_data = {}
    
    def _save_data(self):
        """Save profiled data to storage."""
        # Save as pickle for fast loading
        with open(self.data_path, 'wb') as f:
            pickle.dump(self.profiled_data, f)
        
        # Save as CSV for human readability
        self._save_to_csv()
        
        print(f"Saved {len(self.profiled_data)} profiled operator entries to {self.data_path}")
    
    def _save_to_csv(self):
        """Save profiled data to CSV format."""
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['operator_type', 'batch_size', 'seq_len', 'hidden_dim1', 'hidden_dim2', 
                           'latency_ms', 'memory_footprint_mb', 'flops', 'arithmetic_intensity'])
            
            for key, data in self.profiled_data.items():
                parts = key.split('_')
                if len(parts) >= 5:  # Changed from 6 to 5
                    op_type = parts[0]
                    batch_size = int(parts[1][1:])  # Remove 'b' prefix
                    seq_len = int(parts[2][1:])     # Remove 's' prefix
                    hidden_dim1 = int(parts[3][1:]) # Remove 'h' prefix
                    hidden_dim2 = int(parts[4][1:]) # Remove 'h' prefix
                    
                    writer.writerow([
                        op_type, batch_size, seq_len, hidden_dim1, hidden_dim2,
                        data.get('latency_ms', 0),
                        data.get('memory_footprint_mb', 0),
                        data.get('flops', 0),
                        data.get('arithmetic_intensity', 0)
                    ])
    
    def _generate_query_key(self, op_type: str, batch_size: int, seq_len: int, 
                          hidden_dim1: int, hidden_dim2: int) -> str:
        """Generate a unique query key for an operator."""
        return f"{op_type}_b{batch_size}_s{seq_len}_h{hidden_dim1}_h{hidden_dim2}"
    
    def _benchmark_gemm_operator(self, batch_size: int, seq_len: int, 
                                hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a GEMM operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # Create input tensors
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim1, 
                                 dtype=dtype, device=device)
        weight_tensor = torch.randn(hidden_dim1, hidden_dim2, 
                                  dtype=dtype, device=device)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            _ = torch.matmul(input_tensor, weight_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = torch.matmul(input_tensor, weight_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs
        flops = 2 * batch_size * seq_len * hidden_dim1 * hidden_dim2
        
        # Calculate memory footprint
        input_memory = input_tensor.numel() * input_tensor.element_size()
        weight_memory = weight_tensor.numel() * weight_tensor.element_size()
        output_memory = result.numel() * result.element_size()
        total_memory = input_memory + weight_memory + output_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def _benchmark_attention_operator(self, batch_size: int, seq_len: int, 
                                    hidden_dim: int, num_heads: int) -> Dict[str, float]:
        """Benchmark an attention operator."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        # Validate inputs
        if num_heads is None or num_heads <= 0:
            raise ValueError(f"Invalid num_heads: {num_heads}")
        if hidden_dim is None or hidden_dim <= 0:
            raise ValueError(f"Invalid hidden_dim: {hidden_dim}")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # Create input tensors for attention
        head_dim = hidden_dim // num_heads
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                       dtype=dtype, device=device)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, v)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(head_dim)
            attn_weights = torch.softmax(scores, dim=-1)
            result = torch.matmul(attn_weights, v)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs for attention
        # QK^T: batch_size * num_heads * seq_len * seq_len * head_dim
        qk_flops = batch_size * num_heads * seq_len * seq_len * head_dim
        # Softmax: batch_size * num_heads * seq_len * seq_len
        softmax_flops = batch_size * num_heads * seq_len * seq_len
        # Attention * V: batch_size * num_heads * seq_len * head_dim * seq_len
        attn_v_flops = batch_size * num_heads * seq_len * head_dim * seq_len
        total_flops = qk_flops + softmax_flops + attn_v_flops
        
        # Calculate memory footprint
        q_memory = q.numel() * q.element_size()
        k_memory = k.numel() * k.element_size()
        v_memory = v.numel() * v.element_size()
        scores_memory = batch_size * num_heads * seq_len * seq_len * 4  # float32
        attn_weights_memory = batch_size * num_heads * seq_len * seq_len * 4
        result_memory = result.numel() * result.element_size()
        total_memory = q_memory + k_memory + v_memory + scores_memory + attn_weights_memory + result_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = total_flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': total_flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def profile_gemm_operators(self):
        """Profile all GEMM operators."""
        print("Starting GEMM operator profiling...")
        
        # Generate batch sizes
        if self.configs.batch_size_step == "power-of-2":
            batch_sizes = [2**i for i in range(
                int(np.log2(self.configs.min_batch_size)),
                int(np.log2(self.configs.max_batch_size)) + 1
            )]
        else:
            batch_sizes = list(range(self.configs.min_batch_size, 
                                   self.configs.max_batch_size + 1, 
                                   self.configs.batch_size_step))
        
        # Generate sequence lengths
        if self.configs.seq_len_step == "power-of-2":
            seq_lens = [2**i for i in range(
                int(np.log2(self.configs.min_seq_len)),
                int(np.log2(self.configs.max_seq_len)) + 1
            )]
        else:
            seq_lens = list(range(self.configs.min_seq_len, 
                                self.configs.max_seq_len + 1, 
                                self.configs.seq_len_step))
        
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims) * len(self.configs.gemm_hidden_dims)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim1 in self.configs.gemm_hidden_dims:
                    for hidden_dim2 in self.configs.gemm_hidden_dims:
                        current_op += 1
                        query_key = self._generate_query_key("gemm", batch_size, seq_len, hidden_dim1, hidden_dim2)
                        
                        # Skip if already profiled and not forcing overwrite
                        if query_key in self.profiled_data and not self.configs.force_overwrite:
                            print(f"[{current_op}/{total_ops}] Skipping cached GEMM: {query_key}")
                            continue
                        
                        print(f"[{current_op}/{total_ops}] Profiling GEMM: batch_size={batch_size}, seq_len={seq_len}, "
                              f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}")
                        
                        try:
                            # Clean up memory
                            gc.collect()
                            if TORCH_AVAILABLE:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            # Benchmark the operator
                            metrics = self._benchmark_gemm_operator(batch_size, seq_len, hidden_dim1, hidden_dim2)
                            self.profiled_data[query_key] = metrics
                            
                            print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                                  f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                                  f"FLOPs: {metrics['flops']:.0f}")
                            
                        except Exception as e:
                            print(f"  Error profiling {query_key}: {e}")
                            continue
        
        # Save the profiled data
        self._save_data()
        print("GEMM operator profiling completed!")
    
    def profile_attention_operators(self, num_heads_list: List[int] = [8, 16, 32, 64]):
        """Profile attention operators."""
        print("Starting Attention operator profiling...")
        
        # Generate batch sizes and sequence lengths
        if self.configs.batch_size_step == "power-of-2":
            batch_sizes = [2**i for i in range(
                int(np.log2(self.configs.min_batch_size)),
                int(np.log2(self.configs.max_batch_size)) + 1
            )]
        else:
            batch_sizes = list(range(self.configs.min_batch_size, 
                                   self.configs.max_batch_size + 1, 
                                   self.configs.batch_size_step))
        
        if self.configs.seq_len_step == "power-of-2":
            seq_lens = [2**i for i in range(
                int(np.log2(self.configs.min_seq_len)),
                int(np.log2(self.configs.max_seq_len)) + 1
            )]
        else:
            seq_lens = list(range(self.configs.min_seq_len, 
                                self.configs.max_seq_len + 1, 
                                self.configs.seq_len_step))
        
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims) * len(num_heads_list)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.gemm_hidden_dims:
                    for num_heads in num_heads_list:
                        current_op += 1
                        query_key = self._generate_query_key("attention", batch_size, seq_len, hidden_dim, num_heads)
                        
                        # Skip if already profiled and not forcing overwrite
                        if query_key in self.profiled_data and not self.configs.force_overwrite:
                            print(f"[{current_op}/{total_ops}] Skipping cached Attention: {query_key}")
                            continue
                        
                        print(f"[{current_op}/{total_ops}] Profiling Attention: batch_size={batch_size}, seq_len={seq_len}, "
                              f"hidden_dim={hidden_dim}, num_heads={num_heads}")
                        
                        try:
                            # Clean up memory
                            gc.collect()
                            if TORCH_AVAILABLE:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            # Benchmark the operator
                            metrics = self._benchmark_attention_operator(batch_size, seq_len, hidden_dim, num_heads)
                            self.profiled_data[query_key] = metrics
                            
                            print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                                  f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                                  f"FLOPs: {metrics['flops']:.0f}")
                            
                        except Exception as e:
                            print(f"  Error profiling {query_key}: {e}")
                            continue
        
        # Save the profiled data
        self._save_data()
        print("Attention operator profiling completed!")
    
    def get_operator_latency(self, op_type: str, batch_size: int, seq_len: int, 
                           hidden_dim1: int, hidden_dim2: int) -> Optional[float]:
        """Get the profiled latency for an operator."""
        query_key = self._generate_query_key(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2)
        if query_key in self.profiled_data:
            return self.profiled_data[query_key]['latency_ms']
        return None
    
    def get_operator_metrics(self, op_type: str, batch_size: int, seq_len: int, 
                           hidden_dim1: int, hidden_dim2: int) -> Optional[Dict[str, float]]:
        """Get all profiled metrics for an operator."""
        query_key = self._generate_query_key(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2)
        if query_key in self.profiled_data:
            return self.profiled_data[query_key]
        return None
    
    def interpolate_latency(self, op_type: str, batch_size: int, seq_len: int, 
                          hidden_dim1: int, hidden_dim2: int, max_distance: float = 1000.0) -> Optional[float]:
        """Interpolate latency for an operator using nearest neighbors."""
        # Find the closest profiled operators
        best_match = None
        min_distance = float('inf')
        
        for key, data in self.profiled_data.items():
            if not key.startswith(op_type):
                continue
            
            parts = key.split('_')
            if len(parts) < 6:
                continue
            
            try:
                profiled_batch = int(parts[1][1:])
                profiled_seq = int(parts[2][1:])
                profiled_h1 = int(parts[3][1:])
                profiled_h2 = int(parts[4][1:])
                
                # Calculate distance (weighted by importance)
                distance = (
                    abs(profiled_batch - batch_size) * 0.1 +
                    abs(profiled_seq - seq_len) * 0.1 +
                    abs(profiled_h1 - hidden_dim1) * 1.0 +
                    abs(profiled_h2 - hidden_dim2) * 1.0
                )
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = data['latency_ms']
                    
            except (ValueError, IndexError):
                continue
        
        # Only return interpolation if distance is within threshold
        if min_distance <= max_distance:
            return best_match
        else:
            return None


def main():
    """Main function for running offline profiling."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculon Offline Profiler')
    parser.add_argument('--data_dir', type=str, default='./calculon_offline_data',
                       help='Directory to store profiled data')
    parser.add_argument('--min_batch_size', type=int, default=1,
                       help='Minimum batch size for profiling')
    parser.add_argument('--max_batch_size', type=int, default=32,
                       help='Maximum batch size for profiling')
    parser.add_argument('--min_seq_len', type=int, default=1,
                       help='Minimum sequence length for profiling')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                       help='Maximum sequence length for profiling')
    parser.add_argument('--dtype', type=str, default='float16',
                       choices=['float16', 'float32', 'bfloat16'],
                       help='Data type for profiling')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device for profiling')
    parser.add_argument('--profile_gemm', action='store_true',
                       help='Profile GEMM operators')
    parser.add_argument('--profile_attention', action='store_true',
                       help='Profile attention operators')
    parser.add_argument('--force_overwrite', action='store_true',
                       help='Force overwrite existing profiled data')
    
    args = parser.parse_args()
    
    # Create configuration
    configs = OfflineProfileConfigs(
        data_dir=args.data_dir,
        min_batch_size=args.min_batch_size,
        max_batch_size=args.max_batch_size,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device=args.device,
        force_overwrite=args.force_overwrite
    )
    
    # Create profiler
    profiler = CalculonOfflineProfiler(configs)
    
    # Run profiling
    if args.profile_gemm:
        profiler.profile_gemm_operators()
    
    if args.profile_attention:
        profiler.profile_attention_operators()
    
    if not args.profile_gemm and not args.profile_attention:
        print("No profiling operations specified. Use --profile_gemm or --profile_attention")


if __name__ == "__main__":
    main()



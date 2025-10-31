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
import logging
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
    
    # GEMM profiling dimensions - Expanded to include larger dimensions for better coverage
    gemm_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    
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
        self.log = logging.getLogger(__name__)
        
        # Profiled data storage
        self.profiled_data = {}
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing profiled data from storage."""
        self.log.info(f"[数据加载] 尝试加载离线数据: data_path={self.data_path}")
        if os.path.exists(self.data_path):
            try:
                with open(self.data_path, 'rb') as f:
                    self.profiled_data = pickle.load(f)
                self.log.info(f"[数据加载] 成功加载 {len(self.profiled_data)} 个离线标定算子条目")
                print(f"Loaded {len(self.profiled_data)} profiled operator entries")
            except Exception as e:
                self.log.warning(f"[数据加载] 加载失败: {e}")
                print(f"Failed to load existing data: {e}")
                self.profiled_data = {}
        else:
            self.log.warning(f"[数据加载] 数据文件不存在: data_path={self.data_path}")
            # 检查是否有其他可能的数据文件
            data_dir = os.path.dirname(self.data_path)
            if os.path.exists(data_dir):
                files = [f for f in os.listdir(data_dir) if f.endswith('.pkl')]
                if files:
                    self.log.info(f"[数据加载] 发现目录中存在其他pkl文件: {files}")
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
    
    def _benchmark_layernorm_operator(self, batch_size: int, seq_len: int,
                                      hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a LayerNorm operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # LayerNorm: act_size = batch_size * seq_len * hidden_dim1
        act_size = batch_size * seq_len * hidden_dim1
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim1,
                                 dtype=dtype, device=device)
        
        # LayerNorm with normalized_shape = hidden_dim1
        layernorm = nn.LayerNorm(hidden_dim1, eps=1e-5).to(device).to(dtype)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            _ = layernorm(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = layernorm(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs (forward: 9*act_size, backward: 14*act_size, average: ~11.5*act_size)
        flops = 11 * act_size
        
        # Calculate memory footprint
        input_memory = input_tensor.numel() * input_tensor.element_size()
        weight_memory = layernorm.weight.numel() * layernorm.weight.element_size()
        bias_memory = layernorm.bias.numel() * layernorm.bias.element_size()
        output_memory = result.numel() * result.element_size()
        total_memory = input_memory + weight_memory + bias_memory + output_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def _benchmark_gelu_operator(self, batch_size: int, seq_len: int,
                                 hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a GeLU operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # GeLU: act_size = batch_size * seq_len * hidden_dim1
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim1,
                                 dtype=dtype, device=device)
        gelu = nn.GELU().to(device)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            _ = gelu(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = gelu(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs (forward: 8*act_size, backward: 13*act_size, average: ~10.5*act_size)
        act_size = batch_size * seq_len * hidden_dim1
        flops = 10 * act_size
        
        # Calculate memory footprint
        input_memory = input_tensor.numel() * input_tensor.element_size()
        output_memory = result.numel() * result.element_size()
        total_memory = input_memory + output_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def _benchmark_softmax_operator(self, batch_size: int, seq_len: int,
                                   hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a SoftMax operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # SoftMax: act_size = batch_size * seq_len * hidden_dim1 (or seq_len * seq_len for attention)
        # Use hidden_dim1 as the dimension for softmax
        # If hidden_dim2 > hidden_dim1, it might be for attention matrix (seq_len x seq_len)
        if hidden_dim2 > hidden_dim1 and hidden_dim2 == hidden_dim1 * hidden_dim1:
            # This might be attention matrix softmax: (batch, heads, seq_len, seq_len)
            # For simplicity, use seq_len x seq_len
            input_tensor = torch.randn(batch_size, 1, hidden_dim1, hidden_dim1,
                                     dtype=dtype, device=device)
        else:
            input_tensor = torch.randn(batch_size, seq_len, hidden_dim1,
                                     dtype=dtype, device=device)
        
        softmax = nn.Softmax(dim=-1).to(device)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            _ = softmax(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = softmax(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs (forward: 5*act_size, backward: 8*act_size, average: ~6.5*act_size)
        act_size = input_tensor.numel()
        flops = 6 * act_size
        
        # Calculate memory footprint
        input_memory = input_tensor.numel() * input_tensor.element_size()
        output_memory = result.numel() * result.element_size()
        total_memory = input_memory + output_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def _benchmark_dropout_operator(self, batch_size: int, seq_len: int,
                                   hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a DropOut operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # DropOut: act_size = batch_size * seq_len * hidden_dim1
        input_tensor = torch.randn(batch_size, seq_len, hidden_dim1,
                                 dtype=dtype, device=device)
        dropout = nn.Dropout(p=0.1).to(device)
        
        # Warmup
        dropout.train()  # Enable dropout
        for _ in range(self.configs.num_warmup_steps):
            _ = dropout(input_tensor)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = dropout(input_tensor)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs (forward: act_size, backward: act_size, average: act_size)
        act_size = batch_size * seq_len * hidden_dim1
        flops = act_size
        
        # Calculate memory footprint (includes mask for training)
        input_memory = input_tensor.numel() * input_tensor.element_size()
        output_memory = result.numel() * result.element_size()
        mask_memory = input_tensor.numel() * 1  # bool mask, 1 byte per element
        total_memory = input_memory + output_memory + mask_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def _benchmark_bmm_operator(self, batch_size: int, seq_len: int,
                               hidden_dim1: int, hidden_dim2: int) -> Dict[str, float]:
        """Benchmark a BatchMatMul operator using PyTorch."""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for benchmarking")
        
        device = torch.device(self.configs.device)
        dtype = getattr(torch, self.configs.dtype)
        
        # BatchMatMul: batch matmul between (batch, m, n) and (batch, n, k)
        # For attention: typically (batch, seq_len, hidden_dim1) @ (batch, hidden_dim1, hidden_dim2)
        # Or (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        # Use hidden_dim1 as seq_len or head_dim, hidden_dim2 as the other dimension
        
        # If hidden_dim2 is much larger, it might be seq_len^2 (attention matrix)
        if hidden_dim2 > hidden_dim1 * 10:
            # This might be attention: (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
            # Assume hidden_dim1 = head_dim, hidden_dim2 / hidden_dim1 = seq_len
            seq_dim = max(seq_len, hidden_dim2 // hidden_dim1)
            tensor_a = torch.randn(batch_size, 1, seq_dim, hidden_dim1,
                                  dtype=dtype, device=device)
            tensor_b = torch.randn(batch_size, 1, hidden_dim1, seq_dim,
                                  dtype=dtype, device=device)
        else:
            # Standard batch matmul: (batch, m, n) @ (batch, n, k)
            tensor_a = torch.randn(batch_size, seq_len, hidden_dim1,
                                  dtype=dtype, device=device)
            tensor_b = torch.randn(batch_size, hidden_dim1, hidden_dim2,
                                  dtype=dtype, device=device)
        
        # Warmup
        for _ in range(self.configs.num_warmup_steps):
            _ = torch.bmm(tensor_a, tensor_b)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(self.configs.num_profile_steps):
            result = torch.bmm(tensor_a, tensor_b)
        end_event.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        latency_ms = start_event.elapsed_time(end_event) / self.configs.num_profile_steps
        
        # Calculate FLOPs: batch * 2 * m * n * k
        m, n, k = tensor_a.shape[-2], tensor_a.shape[-1], tensor_b.shape[-1]
        batch = tensor_a.shape[0] * (tensor_a.shape[1] if len(tensor_a.shape) > 3 else 1)
        flops = batch * 2 * m * n * k
        
        # Calculate memory footprint
        a_memory = tensor_a.numel() * tensor_a.element_size()
        b_memory = tensor_b.numel() * tensor_b.element_size()
        result_memory = result.numel() * result.element_size()
        total_memory = a_memory + b_memory + result_memory
        memory_footprint_mb = total_memory / (1024 * 1024)
        
        # Calculate arithmetic intensity
        arithmetic_intensity = flops / total_memory if total_memory > 0 else 0
        
        return {
            'latency_ms': latency_ms,
            'memory_footprint_mb': memory_footprint_mb,
            'flops': flops,
            'arithmetic_intensity': arithmetic_intensity
        }
    
    def profile_layernorm_operators(self):
        """Profile all LayerNorm operators."""
        print("Starting LayerNorm operator profiling...")
        
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
        
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.gemm_hidden_dims:
                    current_op += 1
                    query_key = self._generate_query_key("layernorm", batch_size, seq_len, hidden_dim, hidden_dim)
                    
                    if query_key in self.profiled_data and not self.configs.force_overwrite:
                        print(f"[{current_op}/{total_ops}] Skipping cached LayerNorm: {query_key}")
                        continue
                    
                    print(f"[{current_op}/{total_ops}] Profiling LayerNorm: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    
                    try:
                        gc.collect()
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        metrics = self._benchmark_layernorm_operator(batch_size, seq_len, hidden_dim, hidden_dim)
                        self.profiled_data[query_key] = metrics
                        
                        print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                              f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                              f"FLOPs: {metrics['flops']:.0f}")
                        
                    except Exception as e:
                        print(f"  Error profiling {query_key}: {e}")
                        continue
        
        self._save_data()
        print("LayerNorm operator profiling completed!")
    
    def profile_gelu_operators(self):
        """Profile all GeLU operators."""
        print("Starting GeLU operator profiling...")
        
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
        
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.gemm_hidden_dims:
                    current_op += 1
                    query_key = self._generate_query_key("gelu", batch_size, seq_len, hidden_dim, hidden_dim)
                    
                    if query_key in self.profiled_data and not self.configs.force_overwrite:
                        print(f"[{current_op}/{total_ops}] Skipping cached GeLU: {query_key}")
                        continue
                    
                    print(f"[{current_op}/{total_ops}] Profiling GeLU: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    
                    try:
                        gc.collect()
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        metrics = self._benchmark_gelu_operator(batch_size, seq_len, hidden_dim, hidden_dim)
                        self.profiled_data[query_key] = metrics
                        
                        print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                              f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                              f"FLOPs: {metrics['flops']:.0f}")
                        
                    except Exception as e:
                        print(f"  Error profiling {query_key}: {e}")
                        continue
        
        self._save_data()
        print("GeLU operator profiling completed!")
    
    def profile_softmax_operators(self):
        """Profile all SoftMax operators."""
        print("Starting SoftMax operator profiling...")
        
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
        
        # For SoftMax in attention, we need to handle both regular and attention matrix shapes
        # Regular: (batch, seq_len, hidden_dim)
        # Attention matrix: (batch, heads, seq_len, seq_len) where hidden_dim2 might be seq_len^2
        total_ops = 0
        current_op = 0
        
        # Regular softmax shapes
        regular_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims)
        # Attention matrix softmax: seq_len x seq_len
        attention_matrix_shapes = [(seq, seq) for seq in seq_lens if seq <= 1024]  # Limit to reasonable sizes
        attention_ops = len(batch_sizes) * len(attention_matrix_shapes) * min(4, len(self.configs.gemm_hidden_dims))  # Limit heads
        
        total_ops = regular_ops + attention_ops
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.gemm_hidden_dims:
                    current_op += 1
                    query_key = self._generate_query_key("softmax", batch_size, seq_len, hidden_dim, hidden_dim)
                    
                    if query_key in self.profiled_data and not self.configs.force_overwrite:
                        print(f"[{current_op}/{total_ops}] Skipping cached SoftMax: {query_key}")
                        continue
                    
                    print(f"[{current_op}/{total_ops}] Profiling SoftMax: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    
                    try:
                        gc.collect()
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        metrics = self._benchmark_softmax_operator(batch_size, seq_len, hidden_dim, hidden_dim)
                        self.profiled_data[query_key] = metrics
                        
                        print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                              f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                              f"FLOPs: {metrics['flops']:.0f}")
                        
                    except Exception as e:
                        print(f"  Error profiling {query_key}: {e}")
                        continue
            
            # Profile attention matrix softmax shapes
            for seq_shape in attention_matrix_shapes[:4]:  # Limit to first 4
                for hidden_dim in self.configs.gemm_hidden_dims[:4]:  # Limit dimensions
                    current_op += 1
                    seq_len_attn = seq_shape[0]
                    hidden_dim2_attn = seq_len_attn * seq_len_attn  # seq_len^2 for attention matrix
                    query_key = self._generate_query_key("softmax", batch_size, seq_len_attn, hidden_dim, hidden_dim2_attn)
                    
                    if query_key in self.profiled_data and not self.configs.force_overwrite:
                        continue
                    
                    print(f"[{current_op}/{total_ops}] Profiling SoftMax (attention): batch_size={batch_size}, "
                          f"seq_len={seq_len_attn}, hidden_dim1={hidden_dim}, hidden_dim2={hidden_dim2_attn}")
                    
                    try:
                        gc.collect()
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        metrics = self._benchmark_softmax_operator(batch_size, seq_len_attn, hidden_dim, hidden_dim2_attn)
                        self.profiled_data[query_key] = metrics
                        
                        print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                              f"Memory: {metrics['memory_footprint_mb']:.2f}MB")
                        
                    except Exception as e:
                        print(f"  Error profiling {query_key}: {e}")
                        continue
        
        self._save_data()
        print("SoftMax operator profiling completed!")
    
    def profile_dropout_operators(self):
        """Profile all DropOut operators."""
        print("Starting DropOut operator profiling...")
        
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
        
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim in self.configs.gemm_hidden_dims:
                    current_op += 1
                    query_key = self._generate_query_key("dropout", batch_size, seq_len, hidden_dim, hidden_dim)
                    
                    if query_key in self.profiled_data and not self.configs.force_overwrite:
                        print(f"[{current_op}/{total_ops}] Skipping cached DropOut: {query_key}")
                        continue
                    
                    print(f"[{current_op}/{total_ops}] Profiling DropOut: batch_size={batch_size}, seq_len={seq_len}, hidden_dim={hidden_dim}")
                    
                    try:
                        gc.collect()
                        if TORCH_AVAILABLE:
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                        
                        metrics = self._benchmark_dropout_operator(batch_size, seq_len, hidden_dim, hidden_dim)
                        self.profiled_data[query_key] = metrics
                        
                        print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                              f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                              f"FLOPs: {metrics['flops']:.0f}")
                        
                    except Exception as e:
                        print(f"  Error profiling {query_key}: {e}")
                        continue
        
        self._save_data()
        print("DropOut operator profiling completed!")
    
    def profile_bmm_operators(self):
        """Profile all BatchMatMul operators."""
        print("Starting BatchMatMul operator profiling...")
        
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
        
        # BMM can have various shapes, profile common combinations
        total_ops = len(batch_sizes) * len(seq_lens) * len(self.configs.gemm_hidden_dims) * len(self.configs.gemm_hidden_dims)
        current_op = 0
        
        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                for hidden_dim1 in self.configs.gemm_hidden_dims:
                    for hidden_dim2 in self.configs.gemm_hidden_dims:
                        current_op += 1
                        query_key = self._generate_query_key("bmm", batch_size, seq_len, hidden_dim1, hidden_dim2)
                        
                        if query_key in self.profiled_data and not self.configs.force_overwrite:
                            print(f"[{current_op}/{total_ops}] Skipping cached BMM: {query_key}")
                            continue
                        
                        print(f"[{current_op}/{total_ops}] Profiling BMM: batch_size={batch_size}, seq_len={seq_len}, "
                              f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}")
                        
                        try:
                            gc.collect()
                            if TORCH_AVAILABLE:
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            metrics = self._benchmark_bmm_operator(batch_size, seq_len, hidden_dim1, hidden_dim2)
                            self.profiled_data[query_key] = metrics
                            
                            print(f"  Latency: {metrics['latency_ms']:.3f}ms, "
                                  f"Memory: {metrics['memory_footprint_mb']:.2f}MB, "
                                  f"FLOPs: {metrics['flops']:.0f}")
                            
                        except Exception as e:
                            print(f"  Error profiling {query_key}: {e}")
                            continue
        
        self._save_data()
        print("BatchMatMul operator profiling completed!")
    
    def get_operator_latency(self, op_type: str, batch_size: int, seq_len: int, 
                           hidden_dim1: int, hidden_dim2: int) -> Optional[float]:
        """Get the profiled latency for an operator."""
        query_key = self._generate_query_key(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2)
        self.log.info(f"[离线标定查询] op_type={op_type}, batch_size={batch_size}, seq_len={seq_len}, "
                      f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, query_key={query_key}")
        
        if query_key in self.profiled_data:
            latency_ms = self.profiled_data[query_key]['latency_ms']
            self.log.info(f"[离线标定命中] query_key={query_key}, latency_ms={latency_ms:.3f}")
            return latency_ms
        
        self.log.info(f"[离线标定未命中] query_key={query_key}")
        return None
    
    def get_operator_metrics(self, op_type: str, batch_size: int, seq_len: int, 
                           hidden_dim1: int, hidden_dim2: int) -> Optional[Dict[str, float]]:
        """Get all profiled metrics for an operator."""
        query_key = self._generate_query_key(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2)
        if query_key in self.profiled_data:
            return self.profiled_data[query_key]
        return None
    
    def interpolate_latency(self, op_type: str, batch_size: int, seq_len: int, 
                          hidden_dim1: int, hidden_dim2: int, max_distance: float = 1000.0, 
                          k_neighbors: int = 5) -> Optional[float]:
        """Interpolate latency using K-nearest neighbors with weighted averaging."""
        self.log.info(f"[插值开始] op_type={op_type}, batch_size={batch_size}, seq_len={seq_len}, "
                      f"hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}, max_distance={max_distance}, k={k_neighbors}")
        
        # Calculate adaptive threshold based on query dimensions
        adaptive_threshold = max_distance
        if max_distance > 0:
            max_dim = max(hidden_dim1, hidden_dim2, 1)
            if max_dim > 1024:
                scale_factor = 1.0 + np.log10(max_dim / 1024.0) * 0.5
                adaptive_threshold = max_distance * scale_factor
        
        # Collect all candidate operators with their distances
        candidates = []
        candidates_count = 0
        
        for key, data in self.profiled_data.items():
            if not key.startswith(op_type):
                continue
            
            parts = key.split('_')
            if len(parts) < 5:
                continue
            
            try:
                profiled_batch = int(parts[1][1:])
                profiled_seq = int(parts[2][1:])
                profiled_h1 = int(parts[3][1:])
                profiled_h2 = int(parts[4][1:])
                
                # Improved distance calculation using relative error for large values
                batch_diff = abs(profiled_batch - batch_size)
                seq_diff = abs(profiled_seq - seq_len)
                
                # For hidden dimensions, use relative error (percentage-based)
                h1_relative = abs(profiled_h1 - hidden_dim1) / max(hidden_dim1, profiled_h1, 1)
                h2_relative = abs(profiled_h2 - hidden_dim2) / max(hidden_dim2, profiled_h2, 1)
                
                # Also include absolute difference for small values
                h1_absolute = abs(profiled_h1 - hidden_dim1)
                h2_absolute = abs(profiled_h2 - hidden_dim2)
                
                # Combine relative and absolute: use relative for large values, absolute for small
                avg_h1 = (hidden_dim1 + profiled_h1) / 2.0
                avg_h2 = (hidden_dim2 + profiled_h2) / 2.0
                
                h1_distance = max(h1_absolute, h1_relative * avg_h1)
                h2_distance = max(h2_absolute, h2_relative * avg_h2)
                
                # Weighted distance: batch and seq have lower weight
                distance = (
                    batch_diff * 0.1 +
                    seq_diff * 0.1 +
                    h1_distance * 1.0 +
                    h2_distance * 1.0
                )
                
                candidates_count += 1
                candidates.append({
                    'key': key,
                    'latency_ms': data['latency_ms'],
                    'distance': distance,
                    'batch': profiled_batch,
                    'seq': profiled_seq,
                    'h1': profiled_h1,
                    'h2': profiled_h2
                })
                    
            except (ValueError, IndexError, ZeroDivisionError):
                continue
        
        if not candidates:
            self.log.info(f"[插值失败] 未找到任何候选算子")
            return None
        
        # Sort by distance and get K nearest neighbors
        candidates.sort(key=lambda x: x['distance'])
        k_neighbors_actual = min(k_neighbors, len(candidates))
        k_nearest = candidates[:k_neighbors_actual]
        
        min_distance = k_nearest[0]['distance'] if k_nearest else float('inf')
        
        self.log.info(f"[插值搜索] 搜索到 {candidates_count} 个候选算子, 使用前 {k_neighbors_actual} 个近邻, "
                      f"最小距离={min_distance:.3f}, 自适应阈值={adaptive_threshold:.3f}")
        
        # Check if the nearest neighbor is within threshold
        if min_distance > adaptive_threshold:
            self.log.info(f"[插值失败] 最小距离 {min_distance:.3f} 超过自适应阈值 {adaptive_threshold:.3f}")
            return None
        
        # Weighted interpolation using inverse distance weighting
        # Weight = 1 / (distance + epsilon) to avoid division by zero
        epsilon = 1e-6
        total_weight = 0.0
        weighted_sum = 0.0
        
        for neighbor in k_nearest:
            # Use inverse distance weighting: closer neighbors have more weight
            weight = 1.0 / (neighbor['distance'] + epsilon)
            weighted_sum += neighbor['latency_ms'] * weight
            total_weight += weight
        
        if total_weight == 0:
            self.log.warning(f"[插值失败] 总权重为0，无法进行加权插值")
            return None
        
        interpolated_latency = weighted_sum / total_weight
        
        # Log the K nearest neighbors used
        neighbors_info = ", ".join([f"{n['key']}(dist={n['distance']:.2f})" for n in k_nearest[:3]])
        if len(k_nearest) > 3:
            neighbors_info += f", ... (共{k_neighbors_actual}个)"
        
        self.log.info(f"[插值成功] query=({op_type}, b={batch_size}, s={seq_len}, h1={hidden_dim1}, h2={hidden_dim2}), "
                     f"使用K={k_neighbors_actual}近邻, 最小距离={min_distance:.3f}, "
                     f"插值延迟={interpolated_latency:.3f}ms, 使用的近邻: {neighbors_info}")
        
        return interpolated_latency


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



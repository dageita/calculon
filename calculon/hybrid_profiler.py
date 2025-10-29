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
Hybrid profiler that combines Calculon's theoretical computation with offline profiled data.
This module provides a fusion layer between Calculon's efficiency-based computation and 
offline profiled operator latencies.
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging

from calculon import *
from calculon.offline_profiler import CalculonOfflineProfiler, OfflineProfileConfigs


@dataclass
class HybridProfilerConfigs:
    """Configuration for the hybrid profiler."""
    
    # Offline profiler settings
    offline_data_dir: str = "./calculon_offline_data"
    offline_data_filename: str = "operator_profiles.pkl"
    
    # Fusion strategy
    fusion_strategy: str = "hybrid"  # "offline_only", "calculon_only", "hybrid"
    interpolation_enabled: bool = True
    fallback_to_calculon: bool = True
    
    # Confidence thresholds
    min_confidence_threshold: float = 0.8
    max_interpolation_distance: float = 2.0
    
    # Performance tuning
    enable_caching: bool = True
    cache_size: int = 1000


class HybridProfiler:
    """Hybrid profiler that combines Calculon's theoretical computation with offline profiled data."""
    
    def __init__(self, system: System, configs: HybridProfilerConfigs = None):
        self.system = system
        self.configs = configs or HybridProfilerConfigs()
        self.log = logging.getLogger(__name__)
        
        # Initialize offline profiler
        offline_configs = OfflineProfileConfigs(
            data_dir=self.configs.offline_data_dir,
            data_filename=self.configs.offline_data_filename
        )
        self.offline_profiler = CalculonOfflineProfiler(offline_configs, system)
        
        # Cache for computed results
        self._computation_cache = {} if self.configs.enable_caching else None
        
        # Statistics
        self._stats = {
            'offline_hits': 0,
            'calculon_fallback': 0,
            'interpolation_hits': 0,
            'total_queries': 0,
            'cache_hits': 0,
            'confidence_failures': 0
        }
    
    def _generate_cache_key(self, op_type: str, batch_size: int, seq_len: int, 
                          hidden_dim1: int, hidden_dim2: int, stage: str) -> str:
        """Generate a cache key for the computation."""
        return f"{op_type}_{stage}_b{batch_size}_s{seq_len}_h{hidden_dim1}_h{hidden_dim2}"
    
    def _get_cached_result(self, cache_key: str) -> Optional[float]:
        """Get cached computation result."""
        if self._computation_cache is None:
            return None
        return self._computation_cache.get(cache_key)
    
    def _cache_result(self, cache_key: str, result: float):
        """Cache computation result."""
        if self._computation_cache is not None:
            # Simple LRU-like cache with size limit
            if len(self._computation_cache) >= self.configs.cache_size:
                # Remove oldest entry (simple implementation)
                oldest_key = next(iter(self._computation_cache))
                del self._computation_cache[oldest_key]
            
            self._computation_cache[cache_key] = result
    
    def _calculate_confidence(self, op_type: str, batch_size: int, seq_len: int, 
                            hidden_dim1: int, hidden_dim2: int) -> float:
        """Calculate confidence score for offline profiled data."""
        # Check if exact match exists
        exact_match = self.offline_profiler.get_operator_latency(
            op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
        )
        if exact_match is not None:
            return 1.0
        
        # Check for nearby matches
        if self.configs.interpolation_enabled:
            interpolated = self.offline_profiler.interpolate_latency(
                op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
            )
            if interpolated is not None:
                # Calculate distance-based confidence
                min_distance = self._calculate_min_distance(
                    op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
                )
                confidence = max(0.0, 1.0 - min_distance / self.configs.max_interpolation_distance)
                return confidence
        
        return 0.0
    
    def _calculate_min_distance(self, op_type: str, batch_size: int, seq_len: int, 
                              hidden_dim1: int, hidden_dim2: int) -> float:
        """Calculate minimum distance to profiled operators."""
        min_distance = float('inf')
        
        for key in self.offline_profiler.profiled_data.keys():
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
                
                # Weighted distance calculation
                distance = (
                    abs(profiled_batch - batch_size) * 0.1 +
                    abs(profiled_seq - seq_len) * 0.1 +
                    abs(profiled_h1 - hidden_dim1) * 1.0 +
                    abs(profiled_h2 - hidden_dim2) * 1.0
                )
                
                min_distance = min(min_distance, distance)
                
            except (ValueError, IndexError):
                continue
        
        return min_distance
    
    def _get_offline_latency(self, op_type: str, batch_size: int, seq_len: int, 
                           hidden_dim1: int, hidden_dim2: int) -> Optional[float]:
        """Get latency from offline profiled data."""
        # Try exact match first
        latency_ms = self.offline_profiler.get_operator_latency(
            op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
        )
        if latency_ms is not None:
            # Convert from milliseconds to seconds
            return latency_ms / 1000.0
        
        
        # Try interpolation if enabled
        if self.configs.interpolation_enabled:
            interpolated_ms = self.offline_profiler.interpolate_latency(
                op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
            )
            if interpolated_ms is not None:
                # Convert from milliseconds to seconds
                return interpolated_ms / 1000.0
        
        return None
    
    def _determine_operator_type(self, layer: Layer) -> str:
        """Determine operator type from layer."""
        if isinstance(layer, Linear):
            return "gemm"
        elif "attention" in layer.name.lower():
            return "attention"
        else:
            return "gemm"  # Default to GEMM for most operations
    
    def _extract_dimensions_from_layer(self, layer: Layer, batch_size: int = None, 
                                     seq_len: int = None, hidden_dim1: int = None, 
                                     hidden_dim2: int = None) -> Tuple[int, int, int, int]:
        """Extract dimensions from layer properties."""
        # First, try to get dimensions from HybridLayer attributes
        if hasattr(layer, '_batch_size') and hasattr(layer, '_seq_len') and \
           hasattr(layer, '_hidden_dim1') and hasattr(layer, '_hidden_dim2'):
            layer_batch_size = getattr(layer, '_batch_size', None)
            layer_seq_len = getattr(layer, '_seq_len', None)
            layer_hidden_dim1 = getattr(layer, '_hidden_dim1', None)
            layer_hidden_dim2 = getattr(layer, '_hidden_dim2', None)
            
            if layer_batch_size is not None and layer_seq_len is not None and \
               layer_hidden_dim1 is not None and layer_hidden_dim2 is not None:
                return int(layer_batch_size), int(layer_seq_len), int(layer_hidden_dim1), int(layer_hidden_dim2)
        
        # Fallback to provided parameters or defaults
        default_batch_size = 1 if batch_size is None else batch_size
        default_seq_len = 1 if seq_len is None else seq_len
        default_hidden_dim1 = 512 if hidden_dim1 is None else hidden_dim1
        default_hidden_dim2 = 512 if hidden_dim2 is None else hidden_dim2
        
        # Try to extract from layer attributes
        if hasattr(layer, 'inputs_size') and hasattr(layer, 'output_size'):
            # For linear layers, try to infer dimensions
            if isinstance(layer, Linear):
                # inputs_size = batch_seq * c_in, output_size = batch_seq * c_out
                # We need to make assumptions about batch_seq
                if layer.inputs_size > 0 and layer.output_size > 0:
                    # Assume square matrices for simplicity
                    sqrt_inputs = int(np.sqrt(layer.inputs_size))
                    sqrt_outputs = int(np.sqrt(layer.output_size))
                    if sqrt_inputs > 0 and sqrt_outputs > 0:
                        default_hidden_dim1 = sqrt_inputs
                        default_hidden_dim2 = sqrt_outputs
        
        return default_batch_size, default_seq_len, default_hidden_dim1, default_hidden_dim2
    
    def _get_calculon_latency(self, layer: Layer, stage: str) -> float:
        """Get latency using Calculon's theoretical computation."""
        # Note: calculon_fallback is incremented in the calling method
        # Call the parent class method directly to avoid recursion
        if isinstance(layer, HybridLayer):
            # Get the original Layer class method
            from calculon.llm.layers import Layer as OriginalLayer
            # Create a temporary layer with the same properties but using original Layer class
            temp_layer = OriginalLayer(
                name=layer.name,
                sys=layer.sys,
                fw_flops=layer.fw_flops,
                agrad_flops=layer.agrad_flops,
                wgrad_flops=layer.wgrad_flops,
                inputs_size=layer.inputs_size,
                output_size=layer.output_size,
                activation_space=layer.activation_space,
                activation_grads=layer.activation_grads,
                weight_space=layer.weight_space,
                weight_grads=layer.weight_grads,
                optim_space=layer.optim_space,
                needs_recompute=layer.needs_recompute,
                needs_recomm=layer.needs_recomm,
                activation_reused=layer.activation_reused,
                activation_stored=layer.activation_stored,
                output_stored=layer.output_stored
            )
            return temp_layer.compute_processing_time(stage)
        else:
            return layer.compute_processing_time(stage)
    
    def get_operator_latency(self, layer: Layer, stage: str, 
                           batch_size: int = None, seq_len: int = None,
                           hidden_dim1: int = None, hidden_dim2: int = None) -> float:
        """Get operator latency using hybrid approach."""
        self._stats['total_queries'] += 1
        
        # Determine operator type based on layer type
        op_type = self._determine_operator_type(layer)
        
        # Try to get dimensions from layer if not provided
        if batch_size is None or seq_len is None or hidden_dim1 is None or hidden_dim2 is None:
            batch_size, seq_len, hidden_dim1, hidden_dim2 = self._extract_dimensions_from_layer(
                layer, batch_size, seq_len, hidden_dim1, hidden_dim2
            )
        
        # Debug: Log the query details
        self.log.debug(f"get_operator_latency: {layer.name}, stage={stage}, op_type={op_type}, batch_size={batch_size}, seq_len={seq_len}, hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}")
        
        # Check cache first
        cache_key = self._generate_cache_key(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2, stage)
        cached_result = self._get_cached_result(cache_key)
        if cached_result is not None:
            self._stats['cache_hits'] += 1
            return cached_result
        
        # Apply fusion strategy
        if self.configs.fusion_strategy == "offline_only":
            # Try exact match first
            exact_latency = self.offline_profiler.get_operator_latency(
                op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
            )
            if exact_latency is not None:
                # Exact match found
                self._stats['offline_hits'] += 1
                latency = exact_latency / 1000.0  # Convert to seconds
                self._cache_result(cache_key, latency)
                return latency
            
            # Try interpolation if enabled
            if self.configs.interpolation_enabled:
                interpolated_ms = self.offline_profiler.interpolate_latency(
                    op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
                )
                if interpolated_ms is not None:
                    # Interpolation found
                    self._stats['interpolation_hits'] += 1
                    latency = interpolated_ms / 1000.0  # Convert to seconds
                    self._cache_result(cache_key, latency)
                    return latency
            
            # Fall back to Calculon if enabled
            if self.configs.fallback_to_calculon:
                self._stats['calculon_fallback'] += 1
                latency = self._get_calculon_latency(layer, stage)
                self._cache_result(cache_key, latency)
                return latency
            else:
                raise RuntimeError(f"No offline data available for {op_type} and fallback disabled")
        
        elif self.configs.fusion_strategy == "calculon_only":
            latency = self._get_calculon_latency(layer, stage)
            self._cache_result(cache_key, latency)
            return latency
        
        else:  # hybrid strategy
            # Get confidence score
            confidence = self._calculate_confidence(op_type, batch_size, seq_len, hidden_dim1, hidden_dim2)
            
            if confidence >= self.configs.min_confidence_threshold:
                # Try exact match first
                exact_latency = self.offline_profiler.get_operator_latency(
                    op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
                )
                if exact_latency is not None:
                    # Exact match found
                    self._stats['offline_hits'] += 1
                    latency = exact_latency / 1000.0  # Convert to seconds
                    self._cache_result(cache_key, latency)
                    return latency
                
                # Try interpolation if enabled
                if self.configs.interpolation_enabled:
                    interpolated_ms = self.offline_profiler.interpolate_latency(
                        op_type, batch_size, seq_len, hidden_dim1, hidden_dim2
                    )
                    if interpolated_ms is not None:
                        # Interpolation found
                        self._stats['interpolation_hits'] += 1
                        latency = interpolated_ms / 1000.0  # Convert to seconds
                        self._cache_result(cache_key, latency)
                        return latency
                
                # Confidence was high but no offline data found - fall back to Calculon
                self._stats['calculon_fallback'] += 1
                latency = self._get_calculon_latency(layer, stage)
                self._cache_result(cache_key, latency)
                return latency
            else:
                # Confidence was low, go directly to Calculon
                self._stats['calculon_fallback'] += 1
                latency = self._get_calculon_latency(layer, stage)
                self._cache_result(cache_key, latency)
                return latency
    
    def _determine_operator_type(self, layer: Layer) -> str:
        """Determine operator type from layer instance."""
        # Check for hybrid layers first
        if hasattr(layer, '__class__') and 'Hybrid' in layer.__class__.__name__:
            # For hybrid layers, always return "gemm" to use offline data
            return "gemm"
        
        # Check for original layer types
        if isinstance(layer, Linear):
            return "gemm"
        elif isinstance(layer, LinearOverlapped):
            return "gemm"
        elif isinstance(layer, BatchMatMul):
            return "bmm"
        elif isinstance(layer, LayerNorm):
            return "layernorm"
        elif isinstance(layer, GeLU):
            return "gelu"
        elif isinstance(layer, SoftMax):
            return "softmax"
        elif isinstance(layer, DropOut):
            return "dropout"
        else:
            return "gemm"  # Default to gemm instead of unknown
    
    def _extract_dimensions_from_layer(self, layer: Layer, batch_size: int = None, 
                                     seq_len: int = None, hidden_dim1: int = None, 
                                     hidden_dim2: int = None) -> Tuple[int, int, int, int]:
        """Extract dimensions from layer if not provided."""
        # First, try to get dimensions from HybridLayer attributes
        if hasattr(layer, '_batch_size') and hasattr(layer, '_seq_len') and \
           hasattr(layer, '_hidden_dim1') and hasattr(layer, '_hidden_dim2'):
            layer_batch_size = getattr(layer, '_batch_size', None)
            layer_seq_len = getattr(layer, '_seq_len', None)
            layer_hidden_dim1 = getattr(layer, '_hidden_dim1', None)
            layer_hidden_dim2 = getattr(layer, '_hidden_dim2', None)
            
            if layer_batch_size is not None and layer_seq_len is not None and \
               layer_hidden_dim1 is not None and layer_hidden_dim2 is not None:
                return int(layer_batch_size), int(layer_seq_len), int(layer_hidden_dim1), int(layer_hidden_dim2)
        
        # Fallback to provided parameters or defaults
        default_batch_size = 1 if batch_size is None else batch_size
        default_seq_len = 1 if seq_len is None else seq_len
        default_hidden_dim1 = 512 if hidden_dim1 is None else hidden_dim1
        default_hidden_dim2 = 512 if hidden_dim2 is None else hidden_dim2
        
        # Try to extract from layer attributes
        if hasattr(layer, 'inputs_size') and hasattr(layer, 'output_size'):
            # For linear layers, try to infer dimensions
            if isinstance(layer, Linear):
                # inputs_size = batch_seq * c_in, output_size = batch_seq * c_out
                # We need to make assumptions about batch_seq
                if layer.inputs_size > 0 and layer.output_size > 0:
                    # Assume square matrices for simplicity
                    sqrt_inputs = int(np.sqrt(layer.inputs_size))
                    sqrt_outputs = int(np.sqrt(layer.output_size))
                    if sqrt_inputs > 0 and sqrt_outputs > 0:
                        default_hidden_dim1 = sqrt_inputs
                        default_hidden_dim2 = sqrt_outputs
        
        return default_batch_size, default_seq_len, default_hidden_dim1, default_hidden_dim2
    
    def get_forward_computation_time(self, layers: List[Layer], batch_size: int = None, 
                                   seq_len: int = None) -> float:
        """Get total forward computation time for a list of layers."""
        total_time = 0.0
        
        for layer in layers:
            layer_time = self.get_operator_latency(
                layer, "fw", batch_size, seq_len
            )
            total_time += layer_time
        
        return total_time
    
    def get_backward_computation_time(self, layers: List[Layer], batch_size: int = None, 
                                    seq_len: int = None) -> float:
        """Get total backward computation time for a list of layers."""
        total_agrad_time = 0.0
        total_wgrad_time = 0.0
        
        for layer in layers:
            agrad_time = self.get_operator_latency(
                layer, "agrad", batch_size, seq_len
            )
            wgrad_time = self.get_operator_latency(
                layer, "wgrad", batch_size, seq_len
            )
            total_agrad_time += agrad_time
            total_wgrad_time += wgrad_time
        
        return total_agrad_time + total_wgrad_time
    
    def get_optimization_time(self, layers: List[Layer]) -> float:
        """Get total optimization time for a list of layers."""
        total_time = 0.0
        
        for layer in layers:
            layer_time = self.get_operator_latency(layer, "optim")
            total_time += layer_time
        
        return total_time
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = self._stats.copy()
        if stats['total_queries'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_queries']
            stats['offline_hit_rate'] = stats['offline_hits'] / stats['total_queries']
            stats['calculon_fallback_rate'] = stats['calculon_fallback'] / stats['total_queries']
            stats['interpolation_hit_rate'] = stats['interpolation_hits'] / stats['total_queries']
            stats['confidence_failure_rate'] = stats['confidence_failures'] / stats['total_queries']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['offline_hit_rate'] = 0.0
            stats['calculon_fallback_rate'] = 0.0
            stats['interpolation_hit_rate'] = 0.0
            stats['confidence_failure_rate'] = 0.0
        
        stats['cache_size'] = len(self._computation_cache) if self._computation_cache else 0
        stats['offline_data_size'] = len(self.offline_profiler.profiled_data)
        
        return stats
    
    def clear_cache(self):
        """Clear computation cache."""
        if self._computation_cache:
            self._computation_cache.clear()
        self.log.info("Computation cache cleared")
    
    def reset_statistics(self):
        """Reset profiling statistics."""
        self._stats = {
            'offline_hits': 0,
            'calculon_fallback': 0,
            'interpolation_hits': 0,
            'total_queries': 0
        }
        self.log.info("Statistics reset")


class HybridLayer(Layer):
    """Enhanced Layer class that uses hybrid profiling."""
    
    def __init__(self, name, sys, hybrid_profiler: HybridProfiler, 
                 fw_flops=0, agrad_flops=0, wgrad_flops=0,
                 inputs_size=0, output_size=0, activation_space=0,
                 activation_grads=0, weight_space=0, weight_grads=0,
                 optim_space=0, needs_recompute=False, needs_recomm=False,
                 activation_reused=False, activation_stored=True,
                 output_stored=True, **kwargs):
        
        super().__init__(name, sys, fw_flops, agrad_flops, wgrad_flops,
                        inputs_size, output_size, activation_space,
                        activation_grads, weight_space, weight_grads,
                        optim_space, needs_recompute, needs_recomm,
                        activation_reused, activation_stored, output_stored)
        
        self.hybrid_profiler = hybrid_profiler
        self._batch_size = kwargs.get('batch_size', None)
        self._seq_len = kwargs.get('seq_len', None)
        self._hidden_dim1 = kwargs.get('hidden_dim1', None)
        self._hidden_dim2 = kwargs.get('hidden_dim2', None)
        
        # Store original layer type for communication methods
        self._original_layer_type = kwargs.get('original_layer_type', None)
        self._original_layer = kwargs.get('original_layer', None)
    
    def compute_processing_time(self, stage):
        """Override to use hybrid profiling."""
        # Extract dimensions from layer properties
        batch_size = getattr(self, '_batch_size', None)
        seq_len = getattr(self, '_seq_len', None)
        hidden_dim1 = getattr(self, '_hidden_dim1', None)
        hidden_dim2 = getattr(self, '_hidden_dim2', None)
        
        # If dimensions are not set, try to extract from layer properties
        if batch_size is None or seq_len is None or hidden_dim1 is None or hidden_dim2 is None:
            # Use the hybrid profiler's dimension extraction method
            batch_size, seq_len, hidden_dim1, hidden_dim2 = self.hybrid_profiler._extract_dimensions_from_layer(
                self, batch_size, seq_len, hidden_dim1, hidden_dim2
            )
        
        # Ensure all parameters are simple integers
        batch_size = int(batch_size) if batch_size is not None else 1
        seq_len = int(seq_len) if seq_len is not None else 1
        hidden_dim1 = int(hidden_dim1) if hidden_dim1 is not None else 1
        hidden_dim2 = int(hidden_dim2) if hidden_dim2 is not None else 1
        
        # Debug: Log the query details
        self.hybrid_profiler.log.debug(f"HybridLayer.compute_processing_time: {self.name}, stage={stage}, batch_size={batch_size}, seq_len={seq_len}, hidden_dim1={hidden_dim1}, hidden_dim2={hidden_dim2}")
        
        return self.hybrid_profiler.get_operator_latency(
            self, stage, batch_size, seq_len, hidden_dim1, hidden_dim2
        )
    
    def get_comm_bytes(self, stage, baseblock=True):
        """Proxy to original layer's get_comm_bytes method."""
        if self._original_layer is not None:
            return self._original_layer.get_comm_bytes(stage, baseblock)
        return super().get_comm_bytes(stage, baseblock)
    
    def compute_net_time(self, stage, baseblock=True):
        """Proxy to original layer's compute_net_time method."""
        if self._original_layer is not None:
            return self._original_layer.compute_net_time(stage, baseblock)
        return super().compute_net_time(stage, baseblock)
    
    def get_exposed_net_time(self, stage, baseblock=True):
        """Proxy to original layer's get_exposed_net_time method."""
        if self._original_layer is not None:
            return self._original_layer.get_exposed_net_time(stage, baseblock)
        return super().get_exposed_net_time(stage, baseblock)
    
    def get_required_bandwidth(self, stage, baseblock=True):
        """Proxy to original layer's get_required_bandwidth method."""
        if self._original_layer is not None:
            return self._original_layer.get_required_bandwidth(stage, baseblock)
        return super().get_required_bandwidth(stage, baseblock)


class HybridLinear(HybridLayer):
    """Hybrid Linear layer."""
    
    def __init__(self, name, sys, hybrid_profiler, batch_seq, c_in, c_out,
                 needs_recompute=False, activation_reused=False,
                 activation_stored=True, output_stored=True, **kwargs):
        
        m, n, k = batch_seq, c_in, c_out
        super().__init__(name, sys, hybrid_profiler,
                        fw_flops=2*m*n*k,
                        agrad_flops=2*m*n*k,
                        wgrad_flops=2*m*n*k,
                        inputs_size=m*n,
                        output_size=m*k,
                        weight_space=n*k,
                        weight_grads=n*k,
                        activation_space=m*n,
                        activation_grads=m*k,
                        optim_space=2*n*k,
                        needs_recompute=needs_recompute,
                        activation_reused=activation_reused,
                        activation_stored=activation_stored,
                        output_stored=output_stored,
                        batch_size=1,
                        seq_len=batch_seq,
                        hidden_dim1=n,
                        hidden_dim2=k,
                        **kwargs)
    
    def use_matrix_engine(self):
        return True



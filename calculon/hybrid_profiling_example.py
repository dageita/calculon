#!/usr/bin/env python3
"""
Example script demonstrating the usage of Calculon's hybrid profiling system.
This script shows how to use the offline profiler and hybrid LLM for improved
computation time estimation.
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

# Add calculon to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from calculon import *
from calculon.offline_profiler import OfflineProfileConfigs, CalculonOfflineProfiler
from calculon.hybrid_profiler import HybridProfilerConfigs
from calculon.hybrid_llm import HybridLlm, create_hybrid_llm


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def run_offline_profiling():
    """Run offline profiling to collect operator performance data."""
    print("="*60)
    print("RUNNING OFFLINE PROFILING")
    print("="*60)
    
    # Load model configuration to get appropriate profiling ranges
    model_config_path = "models/gpt2-345M.json"
    if not os.path.exists(model_config_path):
        model_config_path = "../models/gpt2-345M.json"
    
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
    
    with open(model_config_path, 'r') as f:
        model_config = json.load(f)
    
    # Create offline profiler configuration based on model
    configs = OfflineProfileConfigs(
        data_dir="./calculon_offline_data",
        min_batch_size=1,
        max_batch_size=16,
        min_seq_len=1,
        max_seq_len=model_config.get('seq_size', 1024),
        gemm_hidden_dims=[128, 256, 512, 1024, 2048, model_config.get('hidden', 1024)],
        dtype="float16",
        device="cuda:0" if torch and torch.cuda.is_available() else "cpu",
        num_warmup_steps=3,
        num_profile_steps=5,
        force_overwrite=True
    )
    
    # Create offline profiler
    profiler = CalculonOfflineProfiler(configs)
    
    # Profile GEMM operators
    print("Profiling GEMM operators...")
    profiler.profile_gemm_operators()
    
    # Profile attention operators
    print("Profiling attention operators...")
    attn_heads = model_config.get('attn_heads', 16)
    profiler.profile_attention_operators(num_heads_list=[8, 16, 32, attn_heads])
    
    print("Offline profiling completed!")
    print(f"Profiled data saved to: {configs.data_dir}")
    print(f"Total profiled operators: {len(profiler.profiled_data)}")


def create_sample_model():
    """Create a sample model for testing."""
    # Load system configuration
    system_config_path = "systems/L20.json"
    if not os.path.exists(system_config_path):
        system_config_path = "../systems/L20.json"
    
    if not os.path.exists(system_config_path):
        raise FileNotFoundError(f"System configuration not found: {system_config_path}")
    
    # Load system
    with open(system_config_path, 'r') as f:
        system_config = json.load(f)
    
    sys = System(system_config)
    
    # Load model configuration from gpt2-345M.json
    model_config_path = "models/gpt2-345M.json"
    if not os.path.exists(model_config_path):
        model_config_path = "../models/gpt2-345M.json"
    
    if not os.path.exists(model_config_path):
        raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
    
    # Load model configuration
    with open(model_config_path, 'r') as f:
        app_cfg = json.load(f)
    
    app = Llm.Application(app_cfg)
    
    # Create execution configuration
    exe = Llm.Execution(
        num_procs=1,
        tensor_par=1,
        pipeline_par=1,
        data_par=1,
        tensor_par_net=0,
        pipeline_par_net=0,
        data_par_net=0,
        batch_size=1,
        microbatch_size=1,
        datatype="float16",
        fused_activation=False,
        attention_type="multihead",
        activation_recompute="none",
        pipeline_interleaving=1,
        optimizer_sharding=False,
        tensor_par_comm_type="ar",
        tensor_par_overlap="none",
        seq_par_ag_redo=False,
        data_par_overlap=False,
        weight_offload=False,
        activations_offload=False,
        optimizer_offload=False,
        training=True
    )
    
    # Create logger
    log = logging.getLogger('hybrid_profiling')
    
    return sys, exe, app, log


def run_hybrid_profiling_comparison():
    """Run comparison between original Calculon and hybrid profiling."""
    print("="*60)
    print("RUNNING HYBRID PROFILING COMPARISON")
    print("="*60)
    
    # Create system, execution, app, and log configs
    sys, exe, app, log = create_sample_model()
    
    # Create hybrid profiler configuration
    hybrid_configs = HybridProfilerConfigs(
        offline_data_dir="./calculon_offline_data",
        fusion_strategy="offline_only",  # Force use of offline data only
        interpolation_enabled=True,
        fallback_to_calculon=True,
        min_confidence_threshold=0.01,  # Very low threshold
        max_interpolation_distance=100.0,  # Lower distance threshold for more interpolation
        enable_caching=False  # Disable cache to force offline data usage
    )
    
    # Create hybrid LLM
    print("1. Creating and running Hybrid LLM (using offline profiled data)...")
    hybrid_llm = create_hybrid_llm(app, log, hybrid_configs)
    hybrid_llm.compile(sys, exe)
    print("   Running Hybrid LLM...")
    hybrid_llm.run(sys)
    print("   ✓ Hybrid LLM completed")
    
    # Create original LLM for comparison
    print("\n2. Creating and running Original LLM (using Calculon theoretical model)...")
    original_llm = Llm(app, log)
    original_llm.compile(sys, exe)
    print("   Running Original LLM...")
    original_llm.run(sys)
    print("   ✓ Original LLM completed")
    
    print("Model configurations:")
    print(f"  Model: GPT-2 345M (from gpt2-345M.json)")
    print(f"  Hidden size: {app.hidden}")
    print(f"  Feedforward size: {app.feedforward}")
    print(f"  Sequence length: {app.seq_size}")
    print(f"  Attention heads: {app.attn_heads}")
    print(f"  Attention size: {app.attn_size}")
    print(f"  Number of blocks: {app.num_blocks}")
    print(f"  Pipeline parallel: {exe.pipeline_par}")
    print(f"  Data parallel: {exe.data_par}")
    print(f"  Tensor parallel: {exe.tensor_par}")
    print(f"  Microbatch size: {exe.microbatch_size}")
    print(f"  Number of microbatches: {exe._num_microbatches}")
    print()
    
    # Compare computation times
    print("\n3. COMPUTATION TIME COMPARISON:")
    print("-" * 50)
    print("   Note: The following times are extracted from the completed runs above")
    print("   - Hybrid LLM uses offline profiled data (real hardware measurements)")
    print("   - Original LLM uses Calculon theoretical model (mathematical calculations)")
    print()
    
    # Forward time
    print("   Extracting forward computation times...")
    fw_time_original = original_llm.get_fw_time()
    fw_time_hybrid = hybrid_llm.get_fw_time()
    fw_improvement = ((fw_time_original - fw_time_hybrid) / fw_time_original * 100) if fw_time_original > 0 else 0
    
    print(f"Forward time:")
    print(f"  Original (Calculon theory): {fw_time_original:.6f}s")
    print(f"  Hybrid (offline data):      {fw_time_hybrid:.6f}s")
    print(f"  Improvement: {fw_improvement:.2f}%")
    print()
    
    # Backward time
    print("   Extracting backward computation times...")
    bw_time_original = original_llm.get_bw_time()
    bw_time_hybrid = hybrid_llm.get_bw_time()
    bw_improvement = ((bw_time_original - bw_time_hybrid) / bw_time_original * 100) if bw_time_original > 0 else 0
    
    print(f"Backward time:")
    print(f"  Original (Calculon theory): {bw_time_original:.6f}s")
    print(f"  Hybrid (offline data):      {bw_time_hybrid:.6f}s")
    print(f"  Improvement: {bw_improvement:.2f}%")
    print()
    
    # Optimization time
    optim_time_original = original_llm.get_optim_step_time()
    optim_time_hybrid = hybrid_llm.get_optim_step_time()
    optim_improvement = ((optim_time_original - optim_time_hybrid) / optim_time_original * 100) if optim_time_original > 0 else 0
    
    print(f"Optimization time:")
    print(f"  Original: {optim_time_original:.6f}s")
    print(f"  Hybrid:   {optim_time_hybrid:.6f}s")
    print(f"  Improvement: {optim_improvement:.2f}%")
    print()
    
    # Total computation time
    total_original = fw_time_original + bw_time_original + optim_time_original
    total_hybrid = fw_time_hybrid + bw_time_hybrid + optim_time_hybrid
    total_improvement = ((total_original - total_hybrid) / total_original * 100) if total_original > 0 else 0
    
    print(f"Total computation time:")
    print(f"  Original: {total_original:.6f}s")
    print(f"  Hybrid:   {total_hybrid:.6f}s")
    print(f"  Improvement: {total_improvement:.2f}%")
    print()
    
    # Print hybrid profiling statistics
    print("\n5. HYBRID PROFILING STATISTICS:")
    print("-" * 40)
    print("   Detailed statistics about offline data usage...")
    hybrid_llm.print_hybrid_profiling_summary()
    
    # Compare efficiencies
    print("\n4. EFFICIENCY COMPARISON:")
    print("-" * 30)
    print("   Comparing efficiency metrics between the two approaches...")
    
    comp_eff_original = original_llm.get_compute_efficiency()
    comp_eff_hybrid = hybrid_llm.get_computation_efficiency_hybrid()
    
    print(f"Computation efficiency:")
    print(f"  Original: {comp_eff_original:.4f}")
    print(f"  Hybrid:   {comp_eff_hybrid:.4f}")
    print()
    
    sys_eff_original = original_llm.get_system_efficiency()
    sys_eff_hybrid = hybrid_llm.get_system_efficiency_hybrid()
    
    print(f"System efficiency:")
    print(f"  Original: {sys_eff_original:.4f}")
    print(f"  Hybrid:   {sys_eff_hybrid:.4f}")
    print()
    
    total_eff_original = original_llm.get_total_efficiency()
    total_eff_hybrid = hybrid_llm.get_total_efficiency_hybrid()
    
    print(f"Total efficiency:")
    print(f"  Original: {total_eff_original:.4f}")
    print(f"  Hybrid:   {total_eff_hybrid:.4f}")
    print()


def run_flow_network_comparison():
    """Run flow network time comparison."""
    print("="*60)
    print("RUNNING FLOW NETWORK COMPARISON")
    print("="*60)
    
    # Create system, execution, app, and log configs
    sys, exe, app, log = create_sample_model()
    
    # Create hybrid profiler configuration
    hybrid_configs = HybridProfilerConfigs(
        offline_data_dir="./calculon_offline_data",
        fusion_strategy="hybrid",
        interpolation_enabled=True,
        fallback_to_calculon=True
    )
    
    # Create hybrid LLM
    hybrid_llm = create_hybrid_llm(app, log, hybrid_configs)
    hybrid_llm.compile(sys, exe)
    hybrid_llm.run(sys)
    
    # Create original LLM for comparison
    original_llm = Llm(app, log)
    original_llm.compile(sys, exe)
    original_llm.run(sys)
    
    print("Flow network time comparison:")
    print("-" * 40)
    
    # Get flow network times
    flow_time_original = original_llm.get_total_flow_network_time()
    flow_time_hybrid = hybrid_llm.get_total_flow_network_time_hybrid()
    
    print(f"Original flow network time: {flow_time_original}")
    print(f"Hybrid flow network time: {flow_time_hybrid}")
    
    if isinstance(flow_time_original, (list, tuple)) and isinstance(flow_time_hybrid, (list, tuple)):
        print(f"Original result length: {len(flow_time_original)}")
        print(f"Hybrid result length: {len(flow_time_hybrid)}")
        
        if len(flow_time_hybrid) > len(flow_time_original):
            print("Hybrid result includes additional profiling statistics:")
            print(f"  Offline hit rate: {flow_time_hybrid[-4]:.2%}")
            print(f"  Calculon fallback rate: {flow_time_hybrid[-3]:.2%}")
            print(f"  Interpolation hit rate: {flow_time_hybrid[-2]:.2%}")
            print(f"  Total queries: {flow_time_hybrid[-1]}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Calculon Hybrid Profiling Example')
    parser.add_argument('--run-offline', action='store_true',
                       help='Run offline profiling')
    parser.add_argument('--run-comparison', action='store_true',
                       help='Run hybrid profiling comparison')
    parser.add_argument('--run-flow-network', action='store_true',
                       help='Run flow network comparison')
    parser.add_argument('--run-all', action='store_true',
                       help='Run all examples')
    
    args = parser.parse_args()
    
    setup_logging()
    
    if args.run_all or args.run_offline:
        try:
            run_offline_profiling()
        except Exception as e:
            print(f"Error in offline profiling: {e}")
            import traceback
            traceback.print_exc()
    
    if args.run_all or args.run_comparison:
        try:
            run_hybrid_profiling_comparison()
        except Exception as e:
            print(f"Error in hybrid profiling comparison: {e}")
            import traceback
            traceback.print_exc()
    
    if args.run_all or args.run_flow_network:
        try:
            run_flow_network_comparison()
        except Exception as e:
            print(f"Error in flow network comparison: {e}")
            import traceback
            traceback.print_exc()
    
    if not any([args.run_offline, args.run_comparison, args.run_flow_network, args.run_all]):
        print("No operations specified. Use --help for available options.")
        print("Example: python hybrid_profiling_example.py --run-all")


if __name__ == "__main__":
    main()


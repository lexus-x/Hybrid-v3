import time
import torch
import numpy as np
from fvcore.nn import FlopCountAnalysis
from src.sota_hybrid_vit import build_sota_model
from src.toy_dynamic_vit import build_model as build_dynamic_model

def measure_hardware_metrics(model, device, batch_sizes=[1, 32, 128], img_size=224, iterations=30):
    model.eval()
    results = {}
    
    for bs in batch_sizes:
        print(f"  -> Profiling Batch Size: {bs}")
        dummy_input = torch.randn(bs, 3, img_size, img_size).to(device)
        
        # 1. Reset Peak Memory
        torch.cuda.reset_peak_memory_stats(device)
        
        # 2. Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = model(dummy_input)
                
        # 3. Measure Peak VRAM
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        # 4. Measure Throughput
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            for _ in range(iterations):
                _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize()
        
        total_time_ms = start_event.elapsed_time(end_event)
        time_per_batch = (total_time_ms / 1000.0) / iterations
        images_per_second = bs / time_per_batch
        
        results[bs] = {
            'throughput_img_sec': images_per_second,
            'peak_vram_mb': peak_vram_mb
        }
        
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=========================================================================")
    print("🔬 AToM vs AToM-Lite vs DynamicViT Hardware Benchmark Suite")
    print("=========================================================================")
    
    print("\n[1/3] Building Architectures...")
    dense_model = build_sota_model("dense").to(device)
    atom_absorb_model = build_sota_model("pruned", prune_mode="absorption").to(device)
    atom_lite_model = build_sota_model("pruned", prune_mode="lite").to(device)
    
    # Build DynamicViT with identical architecture shape to SOTA ViT-Small
    dynamic_model = build_dynamic_model(
        "pruned",
        num_classes=10,
        image_size=224,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        prune_layers=(3, 6, 9),
        keep_ratios=(0.75, 0.5, 0.25)
    ).to(device)
    
    dense_params = sum(p.numel() for p in dense_model.parameters())
    atom_params = sum(p.numel() for p in atom_absorb_model.parameters())
    dynamic_params = sum(p.numel() for p in dynamic_model.parameters())
    
    print(f"  Dense Baseline:          {dense_params:,} parameters (0.0% overhead)")
    print(f"  AToM (Absorption):       {atom_params:,} parameters (0.0% overhead)")
    print(f"  AToM-Lite (Dropping):    {atom_params:,} parameters (0.0% overhead)")
    print(f"  DynamicViT (NeurIPS):    {dynamic_params:,} parameters ({(dynamic_params/dense_params - 1)*100:.2f}% overhead)")
    
    print("\n[2/3] Calculating Mathematical FLOPs (fvcore)...")
    dummy_input_single = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        dense_flops = FlopCountAnalysis(dense_model, dummy_input_single)
        atom_absorb_flops = FlopCountAnalysis(atom_absorb_model, dummy_input_single)
        atom_lite_flops = FlopCountAnalysis(atom_lite_model, dummy_input_single)
        dynamic_flops = FlopCountAnalysis(dynamic_model, dummy_input_single)
        
        dense_flops.unsupported_ops_warnings(False)
        atom_absorb_flops.unsupported_ops_warnings(False)
        atom_lite_flops.unsupported_ops_warnings(False)
        dynamic_flops.unsupported_ops_warnings(False)
        
        d_flops_g = dense_flops.total() / 1e9
        aa_flops_g = atom_absorb_flops.total() / 1e9
        al_flops_g = atom_lite_flops.total() / 1e9
        dyn_flops_g = dynamic_flops.total() / 1e9
        
        print(f"  Dense Baseline:          {d_flops_g:.2f} GFLOPs")
        print(f"  AToM (Absorption):       {aa_flops_g:.2f} GFLOPs ({(1 - aa_flops_g/d_flops_g)*100:.1f}% reduction)")
        print(f"  AToM-Lite (Dropping):    {al_flops_g:.2f} GFLOPs ({(1 - al_flops_g/d_flops_g)*100:.1f}% reduction)")
        print(f"  DynamicViT Pruned:       {dyn_flops_g:.2f} GFLOPs ({(1 - dyn_flops_g/d_flops_g)*100:.1f}% reduction)")
        
    print("\n[3/3] Benchmarking Real-World Hardware Metrics...")
    print("Benchmarking Dense Baseline...")
    dense_metrics = measure_hardware_metrics(dense_model, device)
    
    print("\nBenchmarking AToM (Absorption)...")
    atom_absorb_metrics = measure_hardware_metrics(atom_absorb_model, device)
    
    print("\nBenchmarking AToM-Lite (Dropping)...")
    atom_lite_metrics = measure_hardware_metrics(atom_lite_model, device)
    
    print("\nBenchmarking DynamicViT Pruned...")
    dynamic_metrics = measure_hardware_metrics(dynamic_model, device)
    
    print("\n=========================================================================================")
    print("OFFICIAL UPGRADED REPRODUCTION COMPARATIVE REPORT:")
    print("=========================================================================================")
    print(f"{'Metric':<20} | {'Dense Baseline':<16} | {'DynamicViT (NeurIPS)':<20} | {'AToM':<16} | {'AToM-Lite (Ours)':<16}")
    print("-" * 105)
    print(f"{'Parameters':<20} | {dense_params:>13,}   | {dynamic_params:>17,}    | {atom_params:>13,}  | {atom_params:>13,}")
    print(f"{'Mathematical FLOPs':<20} | {d_flops_g:>11.2f} GFLOPs | {dyn_flops_g:>15.2f} GFLOPs   | {aa_flops_g:>11.2f} GFLOPs  | {al_flops_g:>11.2f} GFLOPs")
    
    for bs in dense_metrics.keys():
        print(f"\n--- Batch Size: {bs} ---")
        d_thr = dense_metrics[bs]['throughput_img_sec']
        dyn_thr = dynamic_metrics[bs]['throughput_img_sec']
        aa_thr = atom_absorb_metrics[bs]['throughput_img_sec']
        al_thr = atom_lite_metrics[bs]['throughput_img_sec']
        
        print(f"{'Throughput (img/s)':<20} | {d_thr:>13.1f}    | {dyn_thr:>17.1f}     | {aa_thr:>13.1f}   | {al_thr:>13.1f} (Ours: {al_thr/dyn_thr:.2f}x vs DynamicViT)")
        
        d_vram = dense_metrics[bs]['peak_vram_mb']
        dyn_vram = dynamic_metrics[bs]['peak_vram_mb']
        aa_vram = atom_absorb_metrics[bs]['peak_vram_mb']
        al_vram = atom_lite_metrics[bs]['peak_vram_mb']
        print(f"{'Peak VRAM (MB)':<20} | {d_vram:>13.1f}    | {dyn_vram:>17.1f}     | {aa_vram:>13.1f}   | {al_vram:>13.1f} (Ours: {(1 - al_vram/dyn_vram)*100:+.1f}% vs DynamicViT)")
    print("=========================================================================================")

if __name__ == '__main__':
    main()

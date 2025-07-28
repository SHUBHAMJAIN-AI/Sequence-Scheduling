#!/usr/bin/env python3
"""
Memory Usage Analysis for Prediction Costs

Calculates detailed memory usage per step and creates memory-focused analysis.
"""

import json
import matplotlib.pyplot as plt
import numpy as np

def analyze_memory_usage():
    """Analyze memory usage from the full dataset results."""
    
    # Load results
    with open('/root/sequence_code/Sequence-Scheduling/full_dataset_analysis/prediction_costs_20250724_225551.json', 'r') as f:
        data = json.load(f)
    
    config_data = data['inference_batch_128']
    memory = config_data['memory']
    processing = config_data['processing']
    timing = config_data['timing']
    costs = config_data['costs']
    
    # Calculate memory metrics
    total_batches = timing['total_batches_processed']
    total_gpu_time = timing['total_cuda_time_s']
    avg_samples_per_step = processing['avg_samples_per_step']
    
    # Memory calculations
    peak_memory_gb = memory['gpu_memory_peak_gb']
    initial_memory_gb = memory['initial_gpu_memory_gb']
    final_memory_gb = memory['final_gpu_memory_gb']
    memory_increase_gb = memory['gpu_memory_used_gb']
    
    # Average memory during processing (approximate)
    avg_active_memory = (initial_memory_gb + final_memory_gb) / 2
    
    # Memory per step calculations
    peak_memory_per_step = peak_memory_gb  # Peak memory required for each step
    avg_memory_per_step = avg_active_memory  # Average memory during processing
    memory_per_sample = peak_memory_gb / avg_samples_per_step
    
    # Memory×Time costs
    total_memory_time_cost = peak_memory_gb * total_gpu_time
    memory_time_cost_per_step = total_memory_time_cost / total_batches
    memory_time_cost_per_sample = total_memory_time_cost / processing['total_samples']
    
    print("="*60)
    print("DETAILED MEMORY USAGE ANALYSIS")
    print("="*60)
    
    print(f"\n📊 MEMORY STATISTICS:")
    print(f"  Initial GPU Memory: {initial_memory_gb:.2f} GB")
    print(f"  Final GPU Memory: {final_memory_gb:.2f} GB")
    print(f"  Peak GPU Memory: {peak_memory_gb:.2f} GB")
    print(f"  Memory Increase: {memory_increase_gb:.2f} GB")
    print(f"  Average Active Memory: {avg_active_memory:.2f} GB")
    
    print(f"\n🔢 MEMORY PER STEP:")
    print(f"  Peak Memory per Step: {peak_memory_per_step:.2f} GB")
    print(f"  Average Memory per Step: {avg_memory_per_step:.2f} GB") 
    print(f"  Memory per Sample: {memory_per_sample:.4f} GB ({memory_per_sample*1024:.1f} MB)")
    
    print(f"\n⏱️ MEMORY×TIME COSTS:")
    print(f"  Total Memory×Time: {total_memory_time_cost:.1f} GB×seconds")
    print(f"  Memory×Time per Step: {memory_time_cost_per_step:.2f} GB×seconds")
    print(f"  Memory×Time per Sample: {memory_time_cost_per_sample:.4f} GB×seconds")
    
    print(f"\n📈 SCALING PROJECTIONS:")
    datasets = [1000, 10000, 50000, 100000]
    for dataset_size in datasets:
        steps_needed = dataset_size / avg_samples_per_step
        gpu_time_needed = dataset_size * costs['gpu_seconds_per_sample']
        memory_time_needed = peak_memory_gb * gpu_time_needed
        print(f"  {dataset_size:6,} samples: {memory_time_needed:8.0f} GB×seconds ({steps_needed:3.0f} steps)")
    
    print(f"\n💡 KEY INSIGHTS:")
    print(f"  • Each processing step requires {peak_memory_gb:.1f} GB GPU memory")
    print(f"  • Memory efficiency: {1000/memory_per_sample:.0f} samples per GB")
    print(f"  • Peak memory utilization occurs during batch processing")
    print(f"  • Memory×Time cost scales linearly with dataset size")
    
    # Create memory visualization
    create_memory_visualization(memory, timing, processing, costs)
    
    return {
        'peak_memory_per_step_gb': peak_memory_per_step,
        'avg_memory_per_step_gb': avg_memory_per_step,
        'memory_per_sample_gb': memory_per_sample,
        'memory_time_cost_per_step': memory_time_cost_per_step,
        'memory_time_cost_per_sample': memory_time_cost_per_sample,
        'total_memory_time_cost': total_memory_time_cost
    }

def create_memory_visualization(memory, timing, processing, costs):
    """Create memory-focused visualization."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Memory Usage Breakdown
    memory_types = ['Initial', 'Final', 'Peak']
    memory_values = [
        memory['initial_gpu_memory_gb'],
        memory['final_gpu_memory_gb'], 
        memory['gpu_memory_peak_gb']
    ]
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = ax1.bar(memory_types, memory_values, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('GPU Memory Usage', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Memory (GB)')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, memory_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f} GB', ha='center', va='bottom', fontweight='bold')
    
    # 2. Memory Per Step Analysis
    total_steps = timing['total_batches_processed']
    avg_samples_per_step = processing['avg_samples_per_step']
    peak_memory = memory['gpu_memory_peak_gb']
    
    step_data = ['Memory per Step', 'Memory per Sample (MB)']
    step_values = [peak_memory, (peak_memory / avg_samples_per_step) * 1024]
    
    ax2.bar(step_data, step_values, color=['orange', 'purple'], alpha=0.8)
    ax2.set_title('Memory Efficiency', fontsize=14, fontweight='bold') 
    ax2.set_ylabel('Memory (GB / MB)')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (label, value) in enumerate(zip(step_data, step_values)):
        unit = 'GB' if i == 0 else 'MB'
        ax2.text(i, value + max(step_values)*0.02, f'{value:.1f} {unit}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 3. Memory×Time Cost Analysis
    total_memory_time = peak_memory * timing['total_cuda_time_s']
    memory_time_per_step = total_memory_time / total_steps
    memory_time_per_sample = total_memory_time / processing['total_samples']
    
    cost_data = ['Total\n(GB×s)', 'Per Step\n(GB×s)', 'Per Sample\n(GB×s)']
    cost_values = [total_memory_time, memory_time_per_step, memory_time_per_sample]
    
    bars = ax3.bar(cost_data, cost_values, color=['red', 'darkred', 'crimson'], alpha=0.8)
    ax3.set_title('Memory×Time Costs', fontsize=14, fontweight='bold')
    ax3.set_ylabel('GB×seconds')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, cost_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(cost_values)*0.02,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Dataset Scaling Memory Costs
    dataset_sizes = [1000, 10000, 50000, 100000]
    memory_costs = []
    
    for size in dataset_sizes:
        gpu_time = size * costs['gpu_seconds_per_sample']
        memory_cost = peak_memory * gpu_time
        memory_costs.append(memory_cost)
    
    ax4.plot(dataset_sizes, memory_costs, 'b-o', linewidth=3, markersize=8)
    ax4.set_title('Memory Cost Scaling', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Dataset Size (samples)')
    ax4.set_ylabel('Memory×Time Cost (GB×s)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    
    # Add annotations
    for x, y in zip(dataset_sizes, memory_costs):
        ax4.annotate(f'{y:.0f}', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    # Save visualization
    output_file = '/root/sequence_code/Sequence-Scheduling/full_dataset_analysis/memory_analysis_visualization.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nMemory visualization saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    results = analyze_memory_usage()
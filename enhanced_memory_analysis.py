#!/usr/bin/env python3
"""
Enhanced Memory Analysis with Per-Step Tracking

Analyzes the new per-step memory tracking data from the full dataset experiment.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os


def load_latest_results():
    """Load the most recent results with per-step memory tracking."""
    results_dir = "/root/sequence_code/Sequence-Scheduling/full_dataset_analysis"
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    latest_file = sorted(json_files)[-1]
    json_path = os.path.join(results_dir, latest_file)
    
    print(f"Loading results from: {json_path}")
    
    with open(json_path, 'r') as f:
        return json.load(f)

def analyze_per_step_memory(results):
    """Analyze the per-step memory tracking data."""
    
    config_data = results['inference_batch_128']
    batch_timings = config_data['timing']['batch_timings']
    step_memory = config_data['step_memory_analysis']
    processing = config_data['processing']
    costs = config_data['costs']
    
    # Create DataFrame for analysis
    df = pd.DataFrame(batch_timings)
    
    print("="*70)
    print("ENHANCED PER-STEP MEMORY ANALYSIS")
    print("="*70)
    
    print(f"\n📊 MEMORY TRACKING SUMMARY:")
    print(f"  Total batches with memory data: {len(df)}")
    print(f"  Memory measurements available: {step_memory['memory_measurements_available']}")
    
    if step_memory['memory_measurements_available']:
        print(f"\n🔢 ACTUAL MEMORY PER STEP:")
        print(f"  Average memory per step: {step_memory['avg_memory_per_step_gb']:.2f} GB")
        print(f"  Min memory per step: {step_memory['min_memory_per_step_gb']:.2f} GB")
        print(f"  Max memory per step: {step_memory['max_memory_per_step_gb']:.2f} GB")
        print(f"  Memory range: {step_memory['max_memory_per_step_gb'] - step_memory['min_memory_per_step_gb']:.2f} GB")
        print(f"  Standard deviation: {df['memory_after_batch_gb'].std():.2f} GB")
        print(f"  Memory per sample: {step_memory['memory_per_sample_gb']:.4f} GB ({step_memory['memory_per_sample_gb']*1024:.1f} MB)")
        
        print(f"\n📈 MEMORY RESERVED (ALLOCATION):")
        print(f"  Average reserved per step: {step_memory['avg_memory_reserved_per_step_gb']:.2f} GB")
        print(f"  Reserved memory efficiency: {step_memory['avg_memory_per_step_gb']/step_memory['avg_memory_reserved_per_step_gb']*100:.1f}%")
        
        print(f"\n⚡ MEMORY PATTERNS:")
        # Analyze memory patterns
        memory_values = df['memory_after_batch_gb'].values
        first_half_avg = np.mean(memory_values[:len(memory_values)//2])
        second_half_avg = np.mean(memory_values[len(memory_values)//2:])
        
        print(f"  First half average: {first_half_avg:.2f} GB")
        print(f"  Second half average: {second_half_avg:.2f} GB")
        print(f"  Memory trend: {'Increasing' if second_half_avg > first_half_avg else 'Decreasing' if second_half_avg < first_half_avg else 'Stable'}")
        
        # Memory efficiency compared to previous estimate
        print(f"\n🔄 COMPARISON WITH PREVIOUS ESTIMATE:")
        print(f"  Previous estimate (peak): 31.37 GB per step")
        print(f"  Actual average: {step_memory['avg_memory_per_step_gb']:.2f} GB per step")
        print(f"  Efficiency gain: {(31.37 - step_memory['avg_memory_per_step_gb'])/31.37*100:.1f}% less memory needed")
        
        # Cost implications
        print(f"\n💰 REVISED COST CALCULATIONS:")
        actual_memory_time_cost = step_memory['avg_memory_per_step_gb'] * costs['gpu_seconds_per_step']
        print(f"  Actual Memory×Time per step: {actual_memory_time_cost:.2f} GB×seconds")
        print(f"  Actual Memory×Time per sample: {actual_memory_time_cost / processing['avg_samples_per_step']:.4f} GB×seconds")
        
        # Dataset scaling with actual memory
        print(f"\n📊 REVISED SCALING PROJECTIONS:")
        datasets = [1000, 10000, 50000, 100000]
        for dataset_size in datasets:
            steps_needed = dataset_size / processing['avg_samples_per_step']
            gpu_time_needed = dataset_size * costs['gpu_seconds_per_sample']
            actual_memory_time_needed = step_memory['avg_memory_per_step_gb'] * gpu_time_needed
            print(f"  {dataset_size:6,} samples: {actual_memory_time_needed:8.0f} GB×seconds (vs {31.37 * gpu_time_needed:8.0f} estimated)")
    
    return df, step_memory

def create_enhanced_memory_visualization(df, step_memory, output_dir):
    """Create comprehensive memory visualization with per-step data."""
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Memory Usage Over Time (Per Step)
    plt.subplot(3, 3, 1)
    plt.plot(df['batch'], df['memory_after_batch_gb'], 'b-o', linewidth=2, markersize=4)
    plt.title('Actual Memory Usage per Step', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Memory (GB)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df['memory_after_batch_gb'].mean(), color='r', linestyle='--', 
                label=f'Average: {df["memory_after_batch_gb"].mean():.2f} GB')
    plt.legend()
    
    # 2. Memory vs Reserved Memory
    plt.subplot(3, 3, 2)
    plt.plot(df['batch'], df['memory_after_batch_gb'], 'b-', linewidth=2, label='Allocated')
    plt.plot(df['batch'], df['memory_reserved_after_batch_gb'], 'r--', linewidth=2, label='Reserved')
    plt.title('Allocated vs Reserved Memory', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Memory (GB)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Memory Distribution
    plt.subplot(3, 3, 3)
    plt.hist(df['memory_after_batch_gb'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Memory Usage Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Memory (GB)')
    plt.ylabel('Frequency')
    plt.axvline(x=df['memory_after_batch_gb'].mean(), color='r', linestyle='--',
                label=f'Mean: {df["memory_after_batch_gb"].mean():.2f} GB')
    plt.axvline(x=df['memory_after_batch_gb'].median(), color='g', linestyle='--',
                label=f'Median: {df["memory_after_batch_gb"].median():.2f} GB')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Memory vs Processing Time
    plt.subplot(3, 3, 4)
    plt.scatter(df['memory_after_batch_gb'], df['cuda_time_s'], alpha=0.7, color='purple')
    plt.title('Memory vs Processing Time', fontsize=14, fontweight='bold')
    plt.xlabel('Memory (GB)')
    plt.ylabel('Processing Time (s)')
    plt.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(df['memory_after_batch_gb'], df['cuda_time_s'])[0, 1]
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 5. Memory Efficiency (Memory per Sample)
    plt.subplot(3, 3, 5)
    memory_per_sample = df['memory_after_batch_gb'] / df['samples_in_batch'] * 1024  # MB
    plt.plot(df['batch'], memory_per_sample, 'g-o', linewidth=2, markersize=4)
    plt.title('Memory per Sample', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Memory per Sample (MB)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=memory_per_sample.mean(), color='r', linestyle='--',
                label=f'Average: {memory_per_sample.mean():.1f} MB')
    plt.legend()
    
    # 6. Memory Utilization Rate
    plt.subplot(3, 3, 6)
    utilization_rate = df['memory_after_batch_gb'] / df['memory_reserved_after_batch_gb'] * 100
    plt.plot(df['batch'], utilization_rate, 'orange', linewidth=2, marker='s', markersize=4)
    plt.title('Memory Utilization Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Utilization (%)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=utilization_rate.mean(), color='r', linestyle='--',
                label=f'Average: {utilization_rate.mean():.1f}%')
    plt.legend()
    
    # 7. Memory vs Throughput
    plt.subplot(3, 3, 7)
    plt.scatter(df['memory_after_batch_gb'], df['samples_per_second'], alpha=0.7, color='brown')
    plt.title('Memory vs Throughput', fontsize=14, fontweight='bold')
    plt.xlabel('Memory (GB)')
    plt.ylabel('Samples per Second')
    plt.grid(True, alpha=0.3)
    
    # 8. Cumulative Memory×Time Cost
    plt.subplot(3, 3, 8)
    cumulative_memory_cost = np.cumsum(df['memory_after_batch_gb'] * df['cuda_time_s'])
    cumulative_samples = np.cumsum(df['samples_in_batch'])
    plt.plot(cumulative_samples, cumulative_memory_cost, 'purple', linewidth=3)
    plt.title('Cumulative Memory×Time Cost', fontsize=14, fontweight='bold')
    plt.xlabel('Cumulative Samples Processed')
    plt.ylabel('Cumulative Memory×Time (GB×s)')
    plt.grid(True, alpha=0.3)
    
    # 9. Summary Statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    summary_text = f"""
ENHANCED MEMORY ANALYSIS

ACTUAL MEASUREMENTS:
• Avg Memory/Step: {step_memory['avg_memory_per_step_gb']:.2f} GB
• Min Memory/Step: {step_memory['min_memory_per_step_gb']:.2f} GB  
• Max Memory/Step: {step_memory['max_memory_per_step_gb']:.2f} GB
• Memory/Sample: {step_memory['memory_per_sample_gb']*1024:.1f} MB

EFFICIENCY:
• Reserved Memory: {step_memory['avg_memory_reserved_per_step_gb']:.2f} GB
• Utilization Rate: {utilization_rate.mean():.1f}%
• Memory Range: {step_memory['max_memory_per_step_gb'] - step_memory['min_memory_per_step_gb']:.2f} GB

COMPARISON:
• Previous Estimate: 31.37 GB/step
• Actual Average: {step_memory['avg_memory_per_step_gb']:.2f} GB/step
• Efficiency Gain: {(31.37 - step_memory['avg_memory_per_step_gb'])/31.37*100:.1f}%

COST IMPACT:
• Actual Memory×Time: {step_memory['avg_memory_per_step_gb'] * 2.39:.1f} GB×s/step
• Cost Reduction: {(31.37 - step_memory['avg_memory_per_step_gb'])/31.37*100:.1f}%
"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'enhanced_memory_analysis_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nEnhanced memory visualization saved to: {output_file}")
    
    return output_file

def main():
    # Load latest results
    results = load_latest_results()
    
    # Analyze per-step memory
    df, step_memory = analyze_per_step_memory(results)
    
    # Create enhanced visualization
    output_dir = "/root/sequence_code/Sequence-Scheduling/full_dataset_analysis"
    viz_file = create_enhanced_memory_visualization(df, step_memory, output_dir)
    
    print(f"\n✅ Enhanced memory analysis complete!")
    print(f"📊 Visualization: {viz_file}")

if __name__ == "__main__":
    main()
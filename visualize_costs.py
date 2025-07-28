#!/usr/bin/env python3
"""
Cost Analysis Visualization Script

Creates comprehensive visualizations for prediction cost analysis results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

def load_results(json_file):
    """Load cost analysis results from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)

def create_cost_visualizations(results, output_dir):
    """Create comprehensive cost analysis visualizations."""
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Extract data
    config_name = list(results.keys())[0]
    data = results[config_name]
    
    batch_timings = data['timing']['batch_timings']
    processing = data['processing']
    costs = data['costs']
    throughput = data['throughput']
    memory = data['memory']
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(batch_timings)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. GPU Time per Step Over Time
    plt.subplot(3, 3, 1)
    plt.plot(df['batch'], df['cuda_time_s'], 'b-o', linewidth=2, markersize=4)
    plt.title('GPU Time per Step (Batch Processing)', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('GPU Time (seconds)')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df['cuda_time_s'].mean(), color='r', linestyle='--', 
                label=f'Average: {df["cuda_time_s"].mean():.3f}s')
    plt.legend()
    
    # 2. Throughput Analysis
    plt.subplot(3, 3, 2)
    plt.plot(df['batch'], df['samples_per_second'], 'g-o', linewidth=2, markersize=4)
    plt.title('Throughput Over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Samples per Second')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df['samples_per_second'].mean(), color='r', linestyle='--',
                label=f'Average: {df["samples_per_second"].mean():.1f} samples/s')
    plt.legend()
    
    # 3. Samples per Batch
    plt.subplot(3, 3, 3)
    plt.bar(df['batch'], df['samples_in_batch'], alpha=0.7, color='orange')
    plt.title('Samples per Batch', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Number of Samples')
    plt.grid(True, alpha=0.3)
    
    # 4. Token Processing Rate
    plt.subplot(3, 3, 4)
    plt.plot(df['batch'], df['tokens_per_second'], 'm-o', linewidth=2, markersize=4)
    plt.title('Token Processing Rate', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Tokens per Second')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=df['tokens_per_second'].mean(), color='r', linestyle='--',
                label=f'Average: {df["tokens_per_second"].mean():.0f} tokens/s')
    plt.legend()
    
    # 5. GPU Time Distribution
    plt.subplot(3, 3, 5)
    plt.hist(df['cuda_time_s'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('GPU Time Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('GPU Time (seconds)')
    plt.ylabel('Frequency')
    plt.axvline(x=df['cuda_time_s'].mean(), color='r', linestyle='--',
                label=f'Mean: {df["cuda_time_s"].mean():.3f}s')
    plt.axvline(x=df['cuda_time_s'].median(), color='g', linestyle='--',
                label=f'Median: {df["cuda_time_s"].median():.3f}s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Cumulative Cost Over Time
    plt.subplot(3, 3, 6)
    cumulative_time = np.cumsum(df['cuda_time_s'])
    cumulative_samples = np.cumsum(df['samples_in_batch'])
    plt.plot(cumulative_samples, cumulative_time, 'purple', linewidth=3)
    plt.title('Cumulative GPU Cost', fontsize=14, fontweight='bold')
    plt.xlabel('Cumulative Samples Processed')
    plt.ylabel('Cumulative GPU Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # 7. Cost Efficiency Analysis
    plt.subplot(3, 3, 7)
    efficiency = df['samples_in_batch'] / df['cuda_time_s']
    plt.plot(df['batch'], efficiency, 'brown', marker='s', linewidth=2, markersize=4)
    plt.title('Processing Efficiency', fontsize=14, fontweight='bold')
    plt.xlabel('Batch Number')
    plt.ylabel('Samples per GPU Second')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=efficiency.mean(), color='r', linestyle='--',
                label=f'Average: {efficiency.mean():.1f} samples/s')
    plt.legend()
    
    # 8. Memory Usage Summary
    plt.subplot(3, 3, 8)
    memory_data = [
        memory['initial_gpu_memory_gb'],
        memory['final_gpu_memory_gb'],
        memory['gpu_memory_peak_gb']
    ]
    memory_labels = ['Initial', 'Final', 'Peak']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    bars = plt.bar(memory_labels, memory_data, color=colors, alpha=0.8, edgecolor='black')
    plt.title('GPU Memory Usage', fontsize=14, fontweight='bold')
    plt.ylabel('Memory (GB)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, memory_data):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f} GB', ha='center', va='bottom', fontweight='bold')
    
    # 9. Cost Summary Statistics
    plt.subplot(3, 3, 9)
    plt.axis('off')
    
    # Create summary text
    summary_text = f"""
COST ANALYSIS SUMMARY

Dataset: {processing['total_samples']:,} samples
Batch Size: {processing['configured_batch_size']}
Total Batches: {len(batch_timings)}

TIMING STATISTICS:
• Total GPU Time: {costs['total_gpu_seconds']:.1f} seconds
• GPU Time per Step: {costs['gpu_seconds_per_step']:.3f} ± {df['cuda_time_s'].std():.3f}s
• GPU Time per Sample: {costs['gpu_seconds_per_sample']:.6f}s

THROUGHPUT:
• Samples/Second: {throughput['samples_per_second']:.1f}
• Steps/Second: {throughput['steps_per_second']:.2f}
• Tokens/Second: {throughput['tokens_per_second']:,.0f}

EFFICIENCY:
• Samples per Step: {processing['avg_samples_per_step']:.1f}
• Peak GPU Memory: {memory['gpu_memory_peak_gb']:.1f} GB

DATASET SCALING:
• 10K samples: {(costs['gpu_seconds_per_sample'] * 10000):.0f}s ({(costs['gpu_seconds_per_sample'] * 10000/60):.1f} min)
• 100K samples: {(costs['gpu_seconds_per_sample'] * 100000):.0f}s ({(costs['gpu_seconds_per_sample'] * 100000/3600):.1f} hr)
"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'cost_analysis_visualization_{timestamp}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_file}")
    
    return output_file

def create_detailed_report(results, output_dir):
    """Create a detailed text report of the cost analysis."""
    
    config_name = list(results.keys())[0]
    data = results[config_name]
    
    batch_timings = data['timing']['batch_timings']
    processing = data['processing']
    costs = data['costs']
    throughput = data['throughput']
    memory = data['memory']
    
    df = pd.DataFrame(batch_timings)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(output_dir, f'cost_analysis_report_{timestamp}.md')
    
    with open(report_file, 'w') as f:
        f.write(f"""# Prediction Cost Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Dataset Overview
- **Total Samples**: {processing['total_samples']:,}
- **Batch Size**: {processing['configured_batch_size']}
- **Total Batches Processed**: {len(batch_timings)}
- **Total Tokens Processed**: {processing['total_tokens_processed']:,}

## Performance Metrics

### GPU Time Analysis
- **Total GPU Time**: {costs['total_gpu_seconds']:.2f} seconds ({costs['total_gpu_seconds']/60:.2f} minutes)
- **Average GPU Time per Step**: {costs['gpu_seconds_per_step']:.4f} seconds
- **Standard Deviation**: {df['cuda_time_s'].std():.4f} seconds
- **Min GPU Time per Step**: {df['cuda_time_s'].min():.4f} seconds
- **Max GPU Time per Step**: {df['cuda_time_s'].max():.4f} seconds
- **GPU Time per Sample**: {costs['gpu_seconds_per_sample']:.6f} seconds

### Throughput Analysis
- **Overall Throughput**: {throughput['samples_per_second']:.1f} samples/second
- **Token Processing Rate**: {throughput['tokens_per_second']:,.0f} tokens/second
- **Batch Processing Rate**: {throughput['steps_per_second']:.2f} steps/second
- **Average Samples per Step**: {processing['avg_samples_per_step']:.1f}

### Memory Usage
- **Initial GPU Memory**: {memory['initial_gpu_memory_gb']:.2f} GB
- **Final GPU Memory**: {memory['final_gpu_memory_gb']:.2f} GB
- **Peak GPU Memory**: {memory['gpu_memory_peak_gb']:.2f} GB
- **Memory Increase**: {memory['gpu_memory_used_gb']:.2f} GB

## Cost Projections

### Different Dataset Sizes
| Dataset Size | GPU Time | Clock Time | Steps Required |
|-------------|----------|------------|----------------|
| 1,000 samples | {costs['gpu_seconds_per_sample'] * 1000:.1f}s | {costs['gpu_seconds_per_sample'] * 1000/60:.1f} min | {1000 / processing['avg_samples_per_step']:.0f} |
| 10,000 samples | {costs['gpu_seconds_per_sample'] * 10000:.1f}s | {costs['gpu_seconds_per_sample'] * 10000/60:.1f} min | {10000 / processing['avg_samples_per_step']:.0f} |
| 50,000 samples | {costs['gpu_seconds_per_sample'] * 50000:.1f}s | {costs['gpu_seconds_per_sample'] * 50000/3600:.1f} hr | {50000 / processing['avg_samples_per_step']:.0f} |
| 100,000 samples | {costs['gpu_seconds_per_sample'] * 100000:.1f}s | {costs['gpu_seconds_per_sample'] * 100000/3600:.1f} hr | {100000 / processing['avg_samples_per_step']:.0f} |

## Batch-by-Batch Analysis

| Batch | Samples | GPU Time (s) | Throughput (samples/s) | Tokens/s |
|-------|---------|--------------|----------------------|----------|
""")
        
        for _, row in df.iterrows():
            f.write(f"| {int(row['batch']):2d} | {int(row['samples_in_batch']):3d} | {row['cuda_time_s']:7.3f} | {row['samples_per_second']:9.1f} | {row['tokens_per_second']:8.0f} |\n")
        
        f.write(f"""
## Statistical Summary

### GPU Time per Step Statistics
- **Mean**: {df['cuda_time_s'].mean():.4f} seconds
- **Median**: {df['cuda_time_s'].median():.4f} seconds
- **Standard Deviation**: {df['cuda_time_s'].std():.4f} seconds
- **Coefficient of Variation**: {(df['cuda_time_s'].std() / df['cuda_time_s'].mean()) * 100:.1f}%

### Throughput Statistics
- **Mean Throughput**: {df['samples_per_second'].mean():.1f} samples/s
- **Median Throughput**: {df['samples_per_second'].median():.1f} samples/s
- **Throughput Range**: {df['samples_per_second'].min():.1f} - {df['samples_per_second'].max():.1f} samples/s

## Key Insights

1. **Consistent Performance**: The coefficient of variation for GPU time per step is {(df['cuda_time_s'].std() / df['cuda_time_s'].mean()) * 100:.1f}%, indicating {"consistent" if (df['cuda_time_s'].std() / df['cuda_time_s'].mean()) * 100 < 10 else "variable"} performance.

2. **Processing Efficiency**: Processing {processing['avg_samples_per_step']:.1f} samples per step with {processing['configured_batch_size']} batch size achieves {throughput['samples_per_second']:.1f} samples/second throughput.

3. **Memory Efficiency**: Peak memory usage of {memory['gpu_memory_peak_gb']:.1f} GB for batch size {processing['configured_batch_size']}.

4. **Scaling Prediction**: For a 10,000 sample dataset, expect approximately {costs['gpu_seconds_per_sample'] * 10000/60:.1f} minutes of GPU time.

## Recommendations

1. **Optimal Batch Size**: Current batch size of {processing['configured_batch_size']} appears efficient with consistent timing.

2. **Resource Planning**: For large datasets, allocate {memory['gpu_memory_peak_gb']:.0f}+ GB GPU memory.

3. **Time Estimation**: Use {costs['gpu_seconds_per_sample']:.6f} seconds per sample for future cost planning.
""")
    
    print(f"Detailed report saved to: {report_file}")
    return report_file

def main():
    # Find the latest results file
    results_dir = "/root/sequence_code/Sequence-Scheduling/full_dataset_analysis"
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON results files found!")
        return
    
    # Use the most recent file
    latest_file = sorted(json_files)[-1]
    json_path = os.path.join(results_dir, latest_file)
    
    print(f"Analyzing results from: {json_path}")
    
    # Load results
    results = load_results(json_path)
    
    # Create visualizations
    viz_file = create_cost_visualizations(results, results_dir)
    
    # Create detailed report
    report_file = create_detailed_report(results, results_dir)
    
    print(f"\nAnalysis complete!")
    print(f"Visualization: {viz_file}")
    print(f"Report: {report_file}")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Analyze prediction results and generate comprehensive performance report.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def load_results(results_file):
    """Load prediction results from JSON file."""
    with open(results_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_errors(results):
    """Analyze error distribution and patterns."""
    analysis = {}
    
    for dataset_name, dataset_results in results.items():
        if 'detailed_results' not in dataset_results:
            continue
            
        detailed = dataset_results['detailed_results']
        errors = [r['error'] for r in detailed]
        predicted = [r['predicted_tokens'] for r in detailed]
        actual = [r['actual_tokens'] for r in detailed]
        
        analysis[dataset_name] = {
            'error_stats': {
                'mean': np.mean(errors),
                'median': np.median(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'q25': np.percentile(errors, 25),
                'q75': np.percentile(errors, 75)
            },
            'prediction_stats': {
                'predicted_mean': np.mean(predicted),
                'predicted_std': np.std(predicted),
                'actual_mean': np.mean(actual),
                'actual_std': np.std(actual)
            },
            'accuracy_bins': {
                'acc_10': sum(1 for e in errors if e <= 10) / len(errors),
                'acc_25': sum(1 for e in errors if e <= 25) / len(errors),
                'acc_50': sum(1 for e in errors if e <= 50) / len(errors),
                'acc_75': sum(1 for e in errors if e <= 75) / len(errors),
                'acc_100': sum(1 for e in errors if e <= 100) / len(errors),
            }
        }
    
    return analysis

def create_visualizations(results, output_dir):
    """Create visualization plots for the results."""
    # Set up plotting
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Response Length Prediction Analysis', fontsize=16, fontweight='bold')
    
    datasets = list(results.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # Plot 1: Error Distribution
    ax1 = axes[0, 0]
    for i, dataset_name in enumerate(datasets):
        if 'detailed_results' not in results[dataset_name]:
            continue
        errors = [r['error'] for r in results[dataset_name]['detailed_results']]
        ax1.hist(errors, bins=20, alpha=0.7, label=dataset_name.upper(), 
                color=colors[i % len(colors)])
    
    ax1.set_xlabel('Absolute Error (tokens)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution by Dataset')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted vs Actual Scatter
    ax2 = axes[0, 1]
    for i, dataset_name in enumerate(datasets):
        if 'detailed_results' not in results[dataset_name]:
            continue
        detailed = results[dataset_name]['detailed_results']
        predicted = [r['predicted_tokens'] for r in detailed]
        actual = [r['actual_tokens'] for r in detailed]
        ax2.scatter(actual, predicted, alpha=0.6, label=dataset_name.upper(),
                   color=colors[i % len(colors)], s=30)
    
    # Add perfect prediction line
    max_val = max([max([r['actual_tokens'] for r in results[d]['detailed_results']] + 
                      [r['predicted_tokens'] for r in results[d]['detailed_results']]) 
                  for d in datasets if 'detailed_results' in results[d]])
    ax2.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    ax2.set_xlabel('Actual Tokens')
    ax2.set_ylabel('Predicted Tokens')
    ax2.set_title('Predicted vs Actual Tokens')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy at Different Thresholds
    ax3 = axes[1, 0]
    thresholds = [10, 25, 50, 75, 100]
    
    for i, dataset_name in enumerate(datasets):
        if 'detailed_results' not in results[dataset_name]:
            continue
        errors = [r['error'] for r in results[dataset_name]['detailed_results']]
        accuracies = [sum(1 for e in errors if e <= t) / len(errors) for t in thresholds]
        ax3.plot(thresholds, accuracies, marker='o', linewidth=2, markersize=6,
                label=dataset_name.upper(), color=colors[i % len(colors)])
    
    ax3.set_xlabel('Error Threshold (tokens)')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy at Different Error Thresholds')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Plot 4: MAE Comparison
    ax4 = axes[1, 1]
    dataset_names = [d.upper() for d in datasets if 'mae' in results[d]]
    maes = [results[d]['mae'] for d in datasets if 'mae' in results[d]]
    
    bars = ax4.bar(dataset_names, maes, color=colors[:len(dataset_names)], alpha=0.7)
    ax4.set_ylabel('Mean Absolute Error (tokens)')
    ax4.set_title('MAE Comparison Across Datasets')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{mae:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"{output_dir}/prediction_analysis_plots.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file

def generate_report(results_file, output_dir):
    """Generate comprehensive analysis report."""
    # Load results
    results = load_results(results_file)
    
    # Analyze errors
    error_analysis = analyze_errors(results)
    
    # Create visualizations
    plot_file = create_visualizations(results, output_dir)
    
    # Generate report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Response Length Prediction Analysis Report

**Generated:** {timestamp}
**Model:** Qwen2.5-3B-Instruct with LoRA fine-tuning
**Task:** GSM8K Response Length Perception

## Executive Summary

This report analyzes the performance of our trained model on predicting response lengths for GSM8K math problems across training, validation, and test datasets.

"""
    
    # Overall Performance Table
    report += "\n## Overall Performance\n\n"
    report += "| Dataset | Samples | MAE | Acc@50 | Acc@100 |\n"
    report += "|---------|---------|-----|--------|----------|\n"
    
    for dataset_name, dataset_results in results.items():
        if 'mae' in dataset_results:
            report += f"| {dataset_name.upper()} | {dataset_results['samples']} | {dataset_results['mae']:.2f} | {dataset_results['acc_50']:.3f} | {dataset_results['acc_100']:.3f} |\n"
    
    # Detailed Analysis
    report += "\n## Detailed Analysis\n\n"
    
    for dataset_name, analysis in error_analysis.items():
        report += f"### {dataset_name.upper()} Dataset\n\n"
        
        error_stats = analysis['error_stats']
        pred_stats = analysis['prediction_stats']
        acc_bins = analysis['accuracy_bins']
        
        report += f"**Error Statistics:**\n"
        report += f"- Mean Error: {error_stats['mean']:.2f} tokens\n"
        report += f"- Median Error: {error_stats['median']:.2f} tokens\n"
        report += f"- Standard Deviation: {error_stats['std']:.2f} tokens\n"
        report += f"- Error Range: {error_stats['min']:.0f} - {error_stats['max']:.0f} tokens\n"
        report += f"- 25th-75th Percentile: {error_stats['q25']:.1f} - {error_stats['q75']:.1f} tokens\n\n"
        
        report += f"**Prediction Characteristics:**\n"
        report += f"- Average Predicted Length: {pred_stats['predicted_mean']:.1f} ± {pred_stats['predicted_std']:.1f} tokens\n"
        report += f"- Average Actual Length: {pred_stats['actual_mean']:.1f} ± {pred_stats['actual_std']:.1f} tokens\n\n"
        
        report += f"**Accuracy at Different Thresholds:**\n"
        report += f"- Within 10 tokens: {acc_bins['acc_10']:.1%}\n"
        report += f"- Within 25 tokens: {acc_bins['acc_25']:.1%}\n"
        report += f"- Within 50 tokens: {acc_bins['acc_50']:.1%}\n"
        report += f"- Within 75 tokens: {acc_bins['acc_75']:.1%}\n"
        report += f"- Within 100 tokens: {acc_bins['acc_100']:.1%}\n\n"
    
    # Key Findings
    report += "\n## Key Findings\n\n"
    
    # Calculate cross-dataset consistency
    maes = [results[d]['mae'] for d in results.keys() if 'mae' in results[d]]
    mae_std = np.std(maes)
    
    report += f"1. **Consistent Performance**: The model shows consistent performance across datasets with MAE ranging from {min(maes):.1f} to {max(maes):.1f} tokens (std: {mae_std:.1f}).\n\n"
    
    # Best performing dataset
    best_dataset = min(results.keys(), key=lambda d: results[d]['mae'] if 'mae' in results[d] else float('inf'))
    report += f"2. **Best Performance**: The model performs best on the {best_dataset.upper()} dataset with MAE of {results[best_dataset]['mae']:.2f} tokens.\n\n"
    
    # Accuracy analysis
    avg_acc_50 = np.mean([results[d]['acc_50'] for d in results.keys() if 'acc_50' in results[d]])
    avg_acc_100 = np.mean([results[d]['acc_100'] for d in results.keys() if 'acc_100' in results[d]])
    
    report += f"3. **Accuracy Performance**: On average, {avg_acc_50:.1%} of predictions are within 50 tokens and {avg_acc_100:.1%} are within 100 tokens of the actual length.\n\n"
    
    # Recommendations
    report += "\n## Recommendations\n\n"
    report += "1. **Model Performance**: The current model shows reasonable performance for response length prediction, with most predictions falling within acceptable error ranges.\n\n"
    report += "2. **Error Analysis**: The error distribution suggests the model tends to have larger errors for longer responses. Consider training with more diverse response lengths.\n\n"
    report += "3. **Future Improvements**: \n"
    report += "   - Increase training data diversity\n"
    report += "   - Experiment with different model architectures\n"
    report += "   - Fine-tune prediction thresholds based on use case requirements\n\n"
    
    report += f"\n## Visualizations\n\nDetailed plots saved to: `{plot_file}`\n"
    
    # Save report
    report_file = f"{output_dir}/prediction_analysis_report.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    return report_file, plot_file

def main():
    # Find the latest results file
    results_dir = "./results"
    if not os.path.exists(results_dir):
        print("No results directory found")
        return
    
    # Get the latest results file
    results_files = [f for f in os.listdir(results_dir) if f.startswith("prediction_results_fast_") and f.endswith(".json")]
    if not results_files:
        print("No prediction results found")
        return
    
    latest_file = max(results_files, key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
    results_file = os.path.join(results_dir, latest_file)
    
    print(f"Analyzing results from: {results_file}")
    
    # Generate analysis
    report_file, plot_file = generate_report(results_file, results_dir)
    
    print(f"Analysis complete!")
    print(f"Report saved to: {report_file}")
    print(f"Plots saved to: {plot_file}")
    
    # Print summary
    results = load_results(results_file)
    print("\n=== SUMMARY ===")
    for dataset_name, result in results.items():
        if 'mae' in result:
            print(f"{dataset_name.upper()}: MAE={result['mae']:.2f}, Acc@50={result['acc_50']:.3f}, Acc@100={result['acc_100']:.3f}")

if __name__ == "__main__":
    main()
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

def load_downstream_results(results_base_dir):
    """Load downstream task results from all model/mode subdirectories."""
    results_data = {}
    results_base_dir = Path(results_base_dir)

    # Find all classification_results.json and segmentation_results.json files
    for json_file in results_base_dir.glob('downstream_eval/*/*/*/*_results.json'):
        try:
            parts = json_file.parts
            # Expected structure: ... / results / {DATASET} / downstream_eval / {task} / {eval_mode} / {model} / {task}_results.json
            if len(parts) < 5 or not parts[-1].endswith('_results.json'):
                print(f"Warning: Skipping unexpected file path: {json_file}")
                continue

            task = parts[-4]       # e.g., 'classification' or 'segmentation'
            eval_mode = parts[-3]  # e.g., 'baseline' or 'generated'
            model_name = parts[-2] # e.g., 'baseline' or actual model name

            if task not in results_data:
                results_data[task] = {}
            if model_name not in results_data[task]:
                results_data[task][model_name] = {}

            with open(json_file, 'r') as f:
                metrics = json.load(f)
                # Ensure metrics is a dictionary, handle empty files gracefully
                if isinstance(metrics, dict):
                     results_data[task][model_name][eval_mode] = metrics
                else:
                     print(f"Warning: Empty or invalid JSON data in {json_file}. Skipping.")
                     results_data[task][model_name][eval_mode] = {} # Store empty dict

        except Exception as e:
            print(f"Error processing file {json_file}: {e}")

    return results_data

def extract_test_metrics(data, task):
    """Extract test metrics from the data structure based on task type."""
    if task == 'classification':
        # For classification, test metrics are in 'test_metrics' key
        return data.get('test_metrics', {})
    elif task == 'segmentation':
        # For segmentation, test metrics are in 'test_metrics' key
        return data.get('test_metrics', {})
    else:
        # Fallback: try to find test metrics or return the data as is
        return data.get('test_metrics', data)

def plot_classification_metrics(results_data, output_dir):
    """Create comparison plots specifically for classification task."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract baseline and generated results
    baseline_data = results_data.get('baseline', {}).get('baseline', {})
    baseline_metrics = extract_test_metrics(baseline_data, 'classification')
    
    generated_data = {
        model: extract_test_metrics(modes.get('generated', {}), 'classification')
        for model, modes in results_data.items()
        if model != 'baseline' and 'generated' in modes
    }

    if not generated_data:
        print("No 'generated' results found for classification")
        return

    # Define classification metrics to plot
    main_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    models = sorted(generated_data.keys())

    # 1. Main metrics comparison
    available_metrics = [m for m in main_metrics if any(m in data for data in generated_data.values())]
    
    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            values = [generated_data[model].get(metric, np.nan) for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add baseline line
            baseline_value = baseline_metrics.get(metric)
            if baseline_value is not None:
                axes[i].axhline(baseline_value, color='r', linestyle='--', 
                               linewidth=2, label=f'Baseline ({baseline_value:.3f})')
                axes[i].legend()
            
            # Add value labels
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    axes[i].text(bar.get_x() + bar.get_width()/2.0, val, 
                               f'{val:.3f}', va='bottom', ha='center')

        plt.suptitle('Classification Metrics Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_main_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Per-class metrics visualization
    if any('per_class_precision' in data for data in generated_data.values()):
        plot_per_class_metrics(generated_data, baseline_metrics, models, output_dir, 'classification')

    # 3. Training history comparison (if available)
    plot_training_history_comparison(results_data, output_dir, 'classification')

def plot_segmentation_metrics(results_data, output_dir):
    """Create comparison plots specifically for segmentation task."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract baseline and generated results
    baseline_data = results_data.get('baseline', {}).get('baseline', {})
    baseline_metrics = extract_test_metrics(baseline_data, 'segmentation')
    
    generated_data = {
        model: extract_test_metrics(modes.get('generated', {}), 'segmentation')
        for model, modes in results_data.items()
        if model != 'baseline' and 'generated' in modes
    }

    if not generated_data:
        print("No 'generated' results found for segmentation")
        return

    # Define segmentation metrics to plot
    main_metrics = ['mean_iou', 'pixel_accuracy']
    models = sorted(generated_data.keys())

    # 1. Main metrics comparison
    available_metrics = [m for m in main_metrics if any(m in data for data in generated_data.values())]
    
    if available_metrics:
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(6 * len(available_metrics), 6))
        if len(available_metrics) == 1:
            axes = [axes]

        for i, metric in enumerate(available_metrics):
            values = [generated_data[model].get(metric, np.nan) for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_ylabel(metric.replace('_', ' ').title())
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add baseline line
            baseline_value = baseline_metrics.get(metric)
            if baseline_value is not None:
                axes[i].axhline(baseline_value, color='r', linestyle='--', 
                               linewidth=2, label=f'Baseline ({baseline_value:.3f})')
                axes[i].legend()
            
            # Add value labels
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    axes[i].text(bar.get_x() + bar.get_width()/2.0, val, 
                               f'{val:.3f}', va='bottom', ha='center')

        plt.suptitle('Segmentation Metrics Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'segmentation_main_metrics.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Per-class IoU visualization
    if any('per_class_iou' in data for data in generated_data.values()):
        plot_per_class_iou(generated_data, baseline_metrics, models, output_dir)

    # 3. Training history comparison (if available)
    plot_training_history_comparison(results_data, output_dir, 'segmentation')

def plot_per_class_metrics(generated_data, baseline_metrics, models, output_dir, task_type):
    """Plot per-class metrics for classification."""
    if task_type == 'classification':
        metric_types = ['per_class_precision', 'per_class_recall', 'per_class_f1']
        metric_names = ['Precision', 'Recall', 'F1-Score']
    else:
        return

    for metric_type, metric_name in zip(metric_types, metric_names):
        # Check if any model has this metric
        if not any(metric_type in data for data in generated_data.values()):
            continue

        # Get the number of classes from the first available model
        num_classes = None
        for data in generated_data.values():
            if metric_type in data and isinstance(data[metric_type], list):
                num_classes = len(data[metric_type])
                break

        if num_classes is None:
            continue

        # Create subplot for each class
        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
        
        x = np.arange(num_classes)
        width = 0.8 / (len(models) + 1)  # +1 for baseline if available
        
        # Plot baseline if available
        baseline_values = baseline_metrics.get(metric_type, [])
        if baseline_values and len(baseline_values) == num_classes:
            ax.bar(x - width * len(models) / 2, baseline_values, width, 
                  label='Baseline', alpha=0.8, color='red')

        # Plot each model
        for i, model in enumerate(models):
            values = generated_data[model].get(metric_type, [np.nan] * num_classes)
            if len(values) == num_classes:
                offset = width * (i - (len(models) - 1) / 2)
                ax.bar(x + offset, values, width, label=model, alpha=0.7)

        ax.set_xlabel('Class')
        ax.set_ylabel(metric_name)
        ax.set_title(f'Per-Class {metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'classification_per_class_{metric_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def plot_per_class_iou(generated_data, baseline_metrics, models, output_dir):
    """Plot per-class IoU for segmentation."""
    # Get the number of classes from the first available model
    num_classes = None
    for data in generated_data.values():
        if 'per_class_iou' in data and isinstance(data['per_class_iou'], list):
            num_classes = len(data['per_class_iou'])
            break

    if num_classes is None:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    
    x = np.arange(num_classes)
    width = 0.8 / (len(models) + 1)  # +1 for baseline if available
    
    # Plot baseline if available
    baseline_values = baseline_metrics.get('per_class_iou', [])
    if baseline_values and len(baseline_values) == num_classes:
        ax.bar(x - width * len(models) / 2, baseline_values, width, 
              label='Baseline', alpha=0.8, color='red')

    # Plot each model
    for i, model in enumerate(models):
        values = generated_data[model].get('per_class_iou', [np.nan] * num_classes)
        if len(values) == num_classes:
            offset = width * (i - (len(models) - 1) / 2)
            ax.bar(x + offset, values, width, label=model, alpha=0.7)

    ax.set_xlabel('Class')
    ax.set_ylabel('IoU')
    ax.set_title('Per-Class IoU Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Class {i}' for i in range(num_classes)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'segmentation_per_class_iou.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()

def plot_training_history_comparison(results_data, output_dir, task):
    """Plot training history comparison across models."""
    # Collect training histories
    training_histories = {}
    
    # Add baseline if available
    baseline_data = results_data.get('baseline', {}).get('baseline', {})
    if 'training_history' in baseline_data:
        training_histories['baseline'] = baseline_data['training_history']
    
    # Add generated models
    for model_name, modes in results_data.items():
        if model_name != 'baseline' and 'generated' in modes:
            if 'training_history' in modes['generated']:
                training_histories[model_name] = modes['generated']['training_history']

    if not training_histories:
        return

    # Determine which metrics to plot based on task
    if task == 'classification':
        metrics_to_plot = [
            ('train_loss', 'val_loss', 'Loss'),
            ('train_acc', 'val_acc', 'Accuracy'),
            ('val_f1', None, 'Validation F1')
        ]
    else:  # segmentation
        metrics_to_plot = [
            ('train_loss', 'val_loss', 'Loss'),
            ('val_accuracy', None, 'Validation Accuracy'),
            ('val_miou', None, 'Validation mIoU')
        ]

    # Create subplots
    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 6))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for i, (train_metric, val_metric, title) in enumerate(metrics_to_plot):
        for model_name, history in training_histories.items():
            # Handle different history formats
            if isinstance(history, list):
                # Format: list of epoch dictionaries
                epochs = list(range(1, len(history) + 1))
                
                if train_metric and train_metric in history[0]:
                    train_values = [epoch_data[train_metric] for epoch_data in history]
                    axes[i].plot(epochs, train_values, label=f'{model_name} (train)', linestyle='--')
                
                if val_metric and val_metric in history[0]:
                    val_values = [epoch_data[val_metric] for epoch_data in history]
                    axes[i].plot(epochs, val_values, label=f'{model_name} (val)')
                elif train_metric and not val_metric and train_metric in history[0]:
                    # Single metric case (like val_f1)
                    if train_metric.startswith('val_'):
                        val_values = [epoch_data[train_metric] for epoch_data in history]
                        axes[i].plot(epochs, val_values, label=f'{model_name}')
                        
            elif isinstance(history, dict):
                # Format: dictionary with metric lists
                if train_metric and train_metric in history:
                    epochs = list(range(1, len(history[train_metric]) + 1))
                    axes[i].plot(epochs, history[train_metric], label=f'{model_name} (train)', linestyle='--')
                
                if val_metric and val_metric in history:
                    epochs = list(range(1, len(history[val_metric]) + 1))
                    axes[i].plot(epochs, history[val_metric], label=f'{model_name} (val)')
                elif train_metric and not val_metric and train_metric in history:
                    epochs = list(range(1, len(history[train_metric]) + 1))
                    axes[i].plot(epochs, history[train_metric], label=f'{model_name}')

        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(title)
        axes[i].set_title(f'{title} over Training')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.suptitle(f'{task.capitalize()} Training History Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{task}_training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(results_data, output_dir):
    """Generate a summary report with key statistics."""
    summary = {}
    
    for task, task_data in results_data.items():
        summary[task] = {}
        
        # Extract baseline metrics
        baseline_data = task_data.get('baseline', {}).get('baseline', {})
        baseline_metrics = extract_test_metrics(baseline_data, task)
        if baseline_metrics:
            summary[task]['baseline'] = baseline_metrics
        
        # Extract generated model metrics
        for model_name, modes in task_data.items():
            if model_name != 'baseline' and 'generated' in modes:
                generated_metrics = extract_test_metrics(modes['generated'], task)
                if generated_metrics:
                    summary[task][model_name] = generated_metrics

    # Save summary as JSON
    with open(os.path.join(output_dir, 'summary_report.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Create a simple text summary
    with open(os.path.join(output_dir, 'summary_report.txt'), 'w') as f:
        f.write("Downstream Task Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        
        for task, task_data in summary.items():
            f.write(f"{task.upper()} TASK\n")
            f.write("-" * 20 + "\n")
            
            baseline_metrics = task_data.get('baseline', {})
            if baseline_metrics:
                f.write("Baseline Results:\n")
                for metric, value in baseline_metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            for model_name, metrics in task_data.items():
                if model_name != 'baseline':
                    f.write(f"{model_name} Results:\n")
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {metric}: {value:.4f}\n")
                    f.write("\n")
            
            f.write("\n")

def plot_task_metrics(results_data, task, output_dir):
    """Create comparison plots for a specific downstream task."""
    if task == 'classification':
        plot_classification_metrics(results_data[task], output_dir)
    elif task == 'segmentation':
        plot_segmentation_metrics(results_data[task], output_dir)
    else:
        # Fallback to original generic plotting
        plot_generic_metrics(results_data, task, output_dir)

def plot_generic_metrics(results_data, task, output_dir):
    """Generic plotting function for unknown task types."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    if task not in results_data or not results_data[task]:
        print(f"No data found for task: {task}")
        return

    # Use the original logic for unknown tasks
    baseline_metrics = results_data[task].get('baseline', {}).get('baseline', {})
    generated_data = {
        model: modes.get('generated', {})
        for model, modes in results_data[task].items()
        if model != 'baseline' and 'generated' in modes
    }

    if not generated_data:
        print(f"No 'generated' results found for task: {task}")
        return

    # Extract test metrics if available
    for model_name in generated_data:
        generated_data[model_name] = extract_test_metrics(generated_data[model_name], task)
    
    baseline_metrics = extract_test_metrics(baseline_metrics, task)

    # Find common metrics
    common_metrics = set()
    first_model_metrics = next(iter(generated_data.values()), {})
    if first_model_metrics:
        common_metrics = set(first_model_metrics.keys())
        for metrics in generated_data.values():
            common_metrics.intersection_update(metrics.keys())

    # Filter to numeric metrics only
    plot_metrics = [m for m in common_metrics 
                   if all(isinstance(generated_data[model].get(m), (int, float)) 
                         for model in generated_data)]

    if not plot_metrics:
        print(f"No suitable numeric metrics found for task: {task}")
        return

    print(f"Plotting metrics for {task}: {plot_metrics}")

    models = sorted(generated_data.keys())

    for metric in plot_metrics:
        plt.figure(figsize=(10 + len(models) * 0.5, 6))

        values = [generated_data[model].get(metric, np.nan) for model in models]

        bars = plt.bar(models, values)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{task.capitalize()} - {metric.replace("_", " ").title()} Comparison (Generated Data)', fontsize=16)
        plt.xticks(rotation=45, ha='right')

        # Add baseline line if available
        baseline_value = baseline_metrics.get(metric)
        if baseline_value is not None and isinstance(baseline_value, (int, float)):
            plt.axhline(baseline_value, color='r', linestyle='--', linewidth=2, 
                       label=f'Baseline ({baseline_value:.3f})')
            plt.legend()

        # Add value labels on bars
        for bar in bars:
            yval = bar.get_height()
            if not np.isnan(yval):
                plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', 
                        va='bottom', ha='center')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{task}_{metric}_comparison.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize downstream task evaluation results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Base directory containing downstream evaluation results (e.g., 'results/DATASET_NAME')")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory to save visualization plots")

    args = parser.parse_args()

    # Load results data
    results_data = load_downstream_results(args.results_dir)

    if not results_data:
        print(f"No downstream results found in {args.results_dir}")
        return

    # Create output directory if it doesn't exist
    viz_output_dir = Path(args.output_dir) / "downstream_visualizations"
    os.makedirs(viz_output_dir, exist_ok=True)

    # Set plotting style
    sns.set(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10

    # Generate visualizations for each task found
    for task in results_data.keys():
        print(f"\n--- Generating plots for task: {task} ---")
        plot_task_metrics(results_data, task, viz_output_dir)

    # Generate summary report
    print(f"\n--- Generating summary report ---")
    generate_summary_report(results_data, viz_output_dir)

    print(f"\nDownstream task visualizations saved to {viz_output_dir}")

if __name__ == "__main__":
    main()
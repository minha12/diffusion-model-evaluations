# scripts/evaluation/generate_downstream_report.py

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd  # Using pandas for easier data handling

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

def plot_task_metrics(results_data, task, output_dir):
    """Create comparison plots for a specific downstream task."""
    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    if task not in results_data or not results_data[task]:
        print(f"No data found for task: {task}")
        return

    # --- Data Preparation ---
    baseline_metrics = results_data[task].get('baseline', {}).get('baseline', {})
    generated_data = {
        model: modes.get('generated', {})
        for model, modes in results_data[task].items()
        if model != 'baseline' and 'generated' in modes
    }

    if not generated_data:
        print(f"No 'generated' results found for task: {task}")
        # If only baseline exists, maybe plot that? For now, just return.
        if baseline_metrics:
             print(f"Baseline metrics for {task}: {baseline_metrics}")
        return

    # Identify common metrics across generated models for this task
    common_metrics = set()
    first_model_metrics = next(iter(generated_data.values()), {})
    if first_model_metrics:
         common_metrics = set(first_model_metrics.keys())
         for metrics in generated_data.values():
              common_metrics.intersection_update(metrics.keys())

    # Determine primary metrics (customize these based on typical output)
    if task == 'classification':
        plot_metrics = [m for m in ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro'] if m in common_metrics]
    elif task == 'segmentation':
        plot_metrics = [m for m in ['mean_iou', 'mean_dice'] if m in common_metrics]
    else:
        plot_metrics = sorted(list(common_metrics)) # Plot all common if task unknown

    if not plot_metrics:
         print(f"No common metrics suitable for plotting found for task: {task}")
         return

    print(f"Plotting metrics for {task}: {plot_metrics}")

    # --- Plotting ---
    models = sorted(generated_data.keys())

    for metric in plot_metrics:
        plt.figure(figsize=(10 + len(models) * 0.5, 6)) # Adjust size based on model count

        values = [generated_data[model].get(metric, np.nan) for model in models] # Use NaN for missing

        # Create bar chart for generated models
        bars = plt.bar(models, values)
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{task.capitalize()} - {metric.replace("_", " ").title()} Comparison (Generated Data)', fontsize=16)
        plt.xticks(rotation=45, ha='right')

        # Add baseline line if available
        baseline_value = baseline_metrics.get(metric)
        if baseline_value is not None:
            plt.axhline(baseline_value, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline_value:.3f})')
            plt.legend()

        # Add value labels on bars
        for bar in bars:
             yval = bar.get_height()
             if not np.isnan(yval): # Don't label NaN bars
                 plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center') # Adjust position based on value

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

    # Generate visualizations for each task found
    for task in results_data.keys():
        print(f"\n--- Generating plots for task: {task} ---")
        plot_task_metrics(results_data, task, viz_output_dir)

    print(f"\nDownstream task visualizations saved to {viz_output_dir}")

if __name__ == "__main__":
    main()
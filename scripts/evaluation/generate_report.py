import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

def load_metrics(metrics_dir):
    """Load metrics from all model subdirectories."""
    metrics_data = {}
    
    # Find all metrics_results.yaml files
    metrics_dir = Path(metrics_dir)
    for model_dir in metrics_dir.glob('*/'):
        model_name = model_dir.name
        metrics_file = model_dir / 'metrics_results.yaml'
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = yaml.safe_load(f)
                metrics_data[model_name] = metrics
    
    return metrics_data

def create_bar_charts(metrics_data, output_dir):
    """Create individual bar charts for each metric."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all metric keys (excluding std values)
    all_metrics = []
    for metrics in metrics_data.values():
        all_metrics.extend(k for k in metrics.keys() if not k.endswith('_std'))
    all_metrics = sorted(set(all_metrics))
    
    # Set up the style
    sns.set(style="whitegrid")
    
    # Create a bar chart for each metric
    for metric in all_metrics:
        plt.figure(figsize=(10, 6))
        
        # Extract values and model names
        models = []
        values = []
        errors = []
        
        for model, metrics in metrics_data.items():
            if metric in metrics:
                models.append(model)
                values.append(metrics[metric])
                
                # Add error bars if std exists
                std_key = f"{metric}_std"
                if std_key in metrics:
                    errors.append(metrics[std_key])
                else:
                    errors.append(0)
        
        # Create the bar chart
        plt.bar(models, values, yerr=errors, capsize=10)
        plt.title(f'{metric.upper()} Comparison', fontsize=16)
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'{metric}_comparison.png'), dpi=300)
        plt.close()

def create_radar_chart(metrics_data, output_dir):
    """Create a radar chart comparing all models across metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics to plot (exclude std values)
    metrics_to_plot = sorted(set(k for model in metrics_data.values() 
                              for k in model.keys() 
                              if not k.endswith('_std')))
    
    # Number of metrics (categories)
    N = len(metrics_to_plot)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Add lines for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    # Normalize metrics for radar chart
    # First get min and max for each metric
    metric_ranges = {}
    for metric in metrics_to_plot:
        values = [data[metric] for data in metrics_data.values() if metric in data]
        if values:
            # For FID, lower is better, so we invert the normalization
            if metric == 'fid' or metric == 'kid_mean' or metric.startswith('lpips'):
                metric_ranges[metric] = (min(values), max(values), True)  # Third value indicates inversion
            else:
                metric_ranges[metric] = (min(values), max(values), False)
    
    # Draw the radar chart for each model
    for i, (model, metrics) in enumerate(metrics_data.items()):
        values = []
        for metric in metrics_to_plot:
            if metric in metrics:
                min_val, max_val, invert = metric_ranges[metric]
                # Avoid division by zero
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = (metrics[metric] - min_val) / (max_val - min_val)
                    
                # Invert for metrics where lower is better
                if invert:
                    normalized = 1 - normalized
                    
                values.append(normalized)
            else:
                values.append(0)
                
        # Close the loop
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    
    # Draw y-labels (0-1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison Across All Metrics', size=20, y=1.08)
    plt.tight_layout()
    
    # Save the radar chart
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_chart(metrics_data, output_dir):
    """Create a radar chart comparing all models across metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get metrics to plot (exclude std values)
    metrics_to_plot = sorted(set(k for model in metrics_data.values() 
                              for k in model.keys() 
                              if not k.endswith('_std')))
    
    # Number of metrics (categories)
    N = len(metrics_to_plot)
    
    # What will be the angle of each axis in the plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    # Add lines for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    # Normalize metrics for radar chart
    # First get min and max for each metric
    metric_ranges = {}
    for metric in metrics_to_plot:
        values = [data[metric] for data in metrics_data.values() if metric in data]
        if values:
            # For FID, lower is better, so we invert the normalization
            if metric == 'fid' or metric == 'kid_mean' or metric.startswith('lpips'):
                metric_ranges[metric] = (min(values), max(values), True)  # Third value indicates inversion
            else:
                metric_ranges[metric] = (min(values), max(values), False)
    
    # Draw the radar chart for each model
    for i, (model, metrics) in enumerate(metrics_data.items()):
        values = []
        for metric in metrics_to_plot:
            if metric in metrics:
                min_val, max_val, invert = metric_ranges[metric]
                # Avoid division by zero
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = (metrics[metric] - min_val) / (max_val - min_val)
                    
                # Invert for metrics where lower is better
                if invert:
                    normalized = 1 - normalized
                    
                values.append(normalized)
            else:
                values.append(0)
                
        # Close the loop
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    
    # Draw y-labels (0-1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Model Comparison Across All Metrics', size=20, y=1.08)
    plt.tight_layout()
    
    # Save the radar chart
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_grid(metrics_data, output_dir):
    """Create a summary grid with all important metrics."""
    plt.figure(figsize=(16, 12))
    
    # Define key metrics to highlight - replace ssim_mean with ms_ssim_mean
    key_metrics = ['fid', 'kid_mean', 'lpips_mean', 'psnr_mean', 'ms_ssim_mean']
    
    # Create a grid
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # Create bar charts for key metrics
    for i, metric in enumerate(key_metrics):
        # Adjust placement to avoid overlap - MS_SSIM_MEAN would be at position (1,1)
        if i == 4:  # MS_SSIM_MEAN
            ax = plt.subplot(gs[1, 0])  # Place MS_SSIM_MEAN at bottom-left
        else:
            ax = plt.subplot(gs[i//3, i%3])
        
        # Extract values and model names
        models = []
        values = []
        errors = []
        
        for model, metrics in metrics_data.items():
            if metric in metrics:
                models.append(model)
                values.append(metrics[metric])
                
                # Add error bars if std exists
                std_key = f"{metric.replace('_mean', '')}_std"
                if std_key in metrics:
                    errors.append(metrics[std_key])
                else:
                    errors.append(0)
        
        # Create the bar chart
        ax.bar(models, values, yerr=errors, capsize=5)
        ax.set_title(f'{metric.upper()}', fontsize=14)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.tick_params(axis='x', labelsize=8)
    
    # Add a radar chart in the last position, now using only column 2 of row 1
    ax = plt.subplot(gs[1, 1:3], polar=True)
    
    # Copy radar chart code (simplified)
    metrics_to_plot = key_metrics
    N = len(metrics_to_plot)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    # Normalize metrics for radar chart
    metric_ranges = {}
    for metric in metrics_to_plot:
        values = [data[metric] for data in metrics_data.values() if metric in data]
        if values:
            # For FID, lower is better, so we invert the normalization
            if metric == 'fid' or metric == 'kid_mean' or metric.startswith('lpips'):
                metric_ranges[metric] = (min(values), max(values), True)  # Third value indicates inversion
            else:
                metric_ranges[metric] = (min(values), max(values), False)
    
    for i, (model, metrics) in enumerate(metrics_data.items()):
        values = []
        for metric in metrics_to_plot:
            if metric in metrics:
                min_val, max_val, invert = metric_ranges[metric]
                if max_val == min_val:
                    normalized = 0.5
                else:
                    normalized = (metrics[metric] - min_val) / (max_val - min_val)
                    
                if invert:
                    normalized = 1 - normalized
                    
                values.append(normalized)
            else:
                values.append(0)
                
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_ylim(0, 1)
    
    # Add legend
    plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(metrics_data))
    
    plt.suptitle('Model Performance Summary', fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize model evaluation metrics")
    parser.add_argument("--metrics-dir", type=str, required=True, help="Directory containing model metrics")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load metrics data
    metrics_data = load_metrics(args.metrics_dir)
    
    if not metrics_data:
        print("No metrics data found!")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    create_bar_charts(metrics_data, args.output_dir)
    create_radar_chart(metrics_data, args.output_dir)
    create_summary_grid(metrics_data, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
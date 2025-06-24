import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.gridspec import GridSpec

def load_metrics(metrics_dirs, selected_models=None):
    """Load metrics from model subdirectories across multiple metrics directories."""
    metrics_data = {}
    
    # Handle both single directory (backward compatibility) and multiple directories
    if isinstance(metrics_dirs, str):
        metrics_dirs = [metrics_dirs]
    
    # Determine if we're in single or multiple directory mode
    is_single_dir = len(metrics_dirs) == 1
    
    for metrics_dir in metrics_dirs:
        metrics_dir = Path(metrics_dir)
        
        # Find all metrics_results.yaml files
        for model_dir in metrics_dir.glob('*/'):
            model_name = model_dir.name
            
            # Skip models not in selected_models list if filtering is enabled
            if selected_models and model_name not in selected_models:
                continue
                
            metrics_file = model_dir / 'metrics_results.yaml'
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = yaml.safe_load(f)
                
                # For single directory mode, use original naming (backward compatibility)
                if is_single_dir:
                    unique_model_name = model_name
                else:
                    # For multiple directory mode, extract parent folder name for dataset identifier
                    dataset_identifier = metrics_dir.parent.name
                    unique_model_name = f"{model_name}_{dataset_identifier}"
                
                # Handle naming conflicts by adding suffix
                base_name = unique_model_name
                counter = 1
                while unique_model_name in metrics_data:
                    unique_model_name = f"{base_name}_{counter}"
                    counter += 1
                
                metrics_data[unique_model_name] = metrics
                
                print(f"Loaded metrics for {unique_model_name} from {metrics_file}")
    
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

def normalize_radar_values(values, metric, normalization_method='comparative'):
    """
    Normalization methods for better radar chart visualization.
    
    Args:
        values: List of metric values across all models
        metric: Name of the metric
        normalization_method: 'comparative', 'performance_scaled', 'centered', 'theoretical', or 'adaptive'
    
    Returns:
        Normalized values between 0.2 and 1.0 for better visualization
    """
    values = np.array(values)
    
    # Determine if lower is better for this metric
    lower_is_better = (metric == 'fid' or metric == 'kid_mean' or metric.startswith('lpips'))
    
    if normalization_method == 'comparative':
        # Comparative normalization: maps to 0.3-1.0 range to avoid center clustering
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            normalized = np.full_like(values, 0.65)  # Middle-high value for equal performance
        else:
            # Map to 0.3-1.0 range instead of 0-1
            normalized = 0.3 + 0.7 * (values - min_val) / (max_val - min_val)
    
    elif normalization_method == 'performance_scaled':
        # Performance-scaled: use relative performance differences
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            normalized = np.full_like(values, 0.7)
        else:
            # Calculate relative performance (how much better/worse than average)
            mean_val = np.mean(values)
            range_val = max_val - min_val
            
            # Scale based on deviation from mean, but keep in 0.4-1.0 range
            normalized = 0.4 + 0.6 * (values - min_val) / range_val
            
            # Further adjust based on distance from mean to show relative performance
            mean_normalized = 0.4 + 0.6 * (mean_val - min_val) / range_val
            adjustment = 0.1 * (normalized - mean_normalized)
            normalized = np.clip(normalized + adjustment, 0.4, 1.0)
    
    elif normalization_method == 'centered':
        # Centered normalization: center around 0.6 with reasonable spread
        mean_val = np.mean(values)
        std_val = np.std(values)
        if std_val == 0:
            normalized = np.full_like(values, 0.6)
        else:
            # Normalize around mean, then scale to 0.3-0.9 range
            z_scores = (values - mean_val) / std_val
            # Clip to ±2 standard deviations and map to 0.3-0.9
            z_scores = np.clip(z_scores, -2, 2)
            normalized = 0.6 + 0.15 * z_scores  # 0.6 ± 0.3
    
    elif normalization_method == 'theoretical':
        # Theoretical normalization: based on theoretical bounds of metrics
        if metric == 'fid':
            # FID: 0 is perfect, higher is worse. Typical range 0-200, good models < 50
            normalized = np.clip(1 - values / 100, 0.2, 1.0)
        elif metric == 'kid_mean':
            # KID: 0 is perfect, can be negative. Typical range -0.1 to 0.1
            normalized = np.clip(1 - (values + 0.1) / 0.2, 0.2, 1.0)
        elif metric.startswith('lpips'):
            # LPIPS: 0 is perfect, 1 is worst. Typical range 0-1
            normalized = np.clip(1 - values, 0.2, 1.0)
        elif metric.startswith('psnr'):
            # PSNR: higher is better. Typical range 10-50 dB
            normalized = np.clip(values / 50, 0.2, 1.0)
        elif metric.startswith('ms_ssim') or metric.startswith('ssim'):
            # SSIM/MS-SSIM: 1 is perfect, 0 is worst. Typical range 0.5-1.0
            normalized = np.clip(values, 0.2, 1.0)
        elif metric.startswith('uiqi'):
            # UIQI: 1 is perfect, -1 is worst. Typical range 0-1
            normalized = np.clip((values + 1) / 2, 0.2, 1.0)
        else:
            # Default: assume 0-1 range where 1 is better
            normalized = np.clip(values, 0.2, 1.0)
        
        # Don't invert for theoretical bounds as they're already set correctly
        return normalized
    
    elif normalization_method == 'adaptive':
        # Adaptive normalization: chooses best method based on data distribution
        range_val = np.max(values) - np.min(values)
        std_val = np.std(values)
        mean_val = np.mean(values)
        
        # If values are very similar (low variability), use centered approach
        if range_val < 0.1 * mean_val or std_val < 0.05 * mean_val:
            return normalize_radar_values(values, metric, 'centered')
        # If high variability, use comparative approach
        else:
            return normalize_radar_values(values, metric, 'comparative')
    
    else:  # Default to comparative
        min_val, max_val = np.min(values), np.max(values)
        if max_val == min_val:
            normalized = np.full_like(values, 0.65)
        else:
            normalized = 0.3 + 0.7 * (values - min_val) / (max_val - min_val)
    
    # Invert for metrics where lower is better (except theoretical which handles this)
    if lower_is_better and normalization_method != 'theoretical':
        normalized = 1.3 - normalized  # Invert while keeping in 0.3-1.0 range
    
    return normalized

def create_radar_chart(metrics_data, output_dir, normalization_method='comparative'):
    """Create a radar chart comparing all models across metrics with normalization."""
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
    
    # Create figure with subplots for different normalization methods
    fig, axes = plt.subplots(2, 2, figsize=(20, 16), subplot_kw=dict(polar=True))
    axes = axes.flatten()
    
    normalization_methods = ['comparative', 'performance_scaled', 'centered', 'theoretical']
    method_titles = [
        'Comparative Normalization (0.3-1.0 range)',
        'Performance-Scaled Normalization (relative performance)',
        'Centered Normalization (around mean)',
        'Theoretical Bounds Normalization (metric-specific)'
    ]
    
    # Add lines for each model
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    for method_idx, (norm_method, method_title) in enumerate(zip(normalization_methods, method_titles)):
        ax = axes[method_idx]
        
        # Prepare normalization for all metrics
        metric_normalized = {}
        for metric in metrics_to_plot:
            values = [data[metric] for data in metrics_data.values() if metric in data]
            if values:
                metric_normalized[metric] = normalize_radar_values(values, metric, norm_method)
            else:
                metric_normalized[metric] = [0.6] * len(metrics_data)
        
        # Draw the radar chart for each model
        for i, (model, metrics) in enumerate(metrics_data.items()):
            values = []
            for j, metric in enumerate(metrics_to_plot):
                if metric in metrics:
                    # Get the normalized value for this model
                    model_indices = [k for k, (m, _) in enumerate(metrics_data.items()) if m == model]
                    if model_indices and metric in metric_normalized:
                        values.append(metric_normalized[metric][model_indices[0]])
                    else:
                        values.append(0.6)
                else:
                    values.append(0.6)
                    
            # Close the loop
            values += values[:1]
            
            # Plot the model
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        # Fix axis to go in the right order and start at 12 o'clock
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # Draw axis lines for each angle and label
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_to_plot, fontsize=10)
        
        # Adjust y-axis range and labels based on normalization method
        if norm_method in ['comparative', 'performance_scaled']:
            ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.set_yticklabels(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=8)
            ax.set_ylim(0.2, 1.0)
        elif norm_method == 'centered':
            ax.set_yticks([0.3, 0.45, 0.6, 0.75, 0.9])
            ax.set_yticklabels(['0.3', '0.45', '0.6', '0.75', '0.9'], fontsize=8)
            ax.set_ylim(0.2, 1.0)
        else:  # theoretical
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.set_ylim(0.2, 1.0)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        ax.set_title(method_title, size=14, pad=20)
    
    # Add a single legend for all subplots - adjust positioning for more models
    handles, labels = axes[0].get_legend_handles_labels()
    ncol = min(len(metrics_data), 4)  # Limit columns to prevent overflow
    fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=ncol, fontsize=10)
    
    plt.suptitle('Model Comparison with Normalization Methods', fontsize=20, y=0.95)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    
    # Save the radar chart
    plt.savefig(os.path.join(output_dir, 'radar_chart_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a single radar chart with the best normalization method
    create_single_radar_chart(metrics_data, output_dir, 'comparative')

def create_single_radar_chart(metrics_data, output_dir, normalization_method='comparative'):
    """Create a single radar chart with the specified normalization method."""
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
    
    # Prepare normalization for all metrics
    metric_normalized = {}
    for metric in metrics_to_plot:
        values = [data[metric] for data in metrics_data.values() if metric in data]
        if values:
            metric_normalized[metric] = normalize_radar_values(values, metric, normalization_method)
        else:
            metric_normalized[metric] = [0.6] * len(metrics_data)
    
    # Draw the radar chart for each model
    for i, (model, metrics) in enumerate(metrics_data.items()):
        values = []
        for j, metric in enumerate(metrics_to_plot):
            if metric in metrics:
                # Get the normalized value for this model
                model_indices = [k for k, (m, _) in enumerate(metrics_data.items()) if m == model]
                if model_indices and metric in metric_normalized:
                    values.append(metric_normalized[metric][model_indices[0]])
                else:
                    values.append(0.6)
            else:
                values.append(0.6)
                
        # Close the loop
        values += values[:1]
        
        # Plot the model
        ax.plot(angles, values, linewidth=3, linestyle='solid', label=model, color=colors[i], marker='o', markersize=6)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Fix axis to go in the right order and start at 12 o'clock
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw axis lines for each angle and label
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=12)
    
    # Set appropriate y-axis range and labels
    if normalization_method in ['comparative', 'performance_scaled']:
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_yticklabels(['0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'], fontsize=10)
        ax.set_ylim(0.2, 1.0)
    elif normalization_method == 'centered':
        ax.set_yticks([0.3, 0.45, 0.6, 0.75, 0.9])
        ax.set_yticklabels(['0.3', '0.45', '0.6', '0.75', '0.9'], fontsize=10)
        ax.set_ylim(0.2, 1.0)
    else:  # theoretical
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.set_ylim(0.2, 1.0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend - adjust for many models
    if len(metrics_data) <= 6:
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=11)
    else:
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize=9)
    
    method_names = {
        'comparative': 'Comparative Normalization',
        'performance_scaled': 'Performance-Scaled Normalization',
        'centered': 'Centered Normalization',
        'theoretical': 'Theoretical Bounds Normalization'
    }
    
    plt.title(f'Model Comparison - {method_names.get(normalization_method, normalization_method)}', 
              size=16, y=1.08)
    plt.tight_layout()
    
    # Save the single radar chart
    plt.savefig(os.path.join(output_dir, f'radar_chart_{normalization_method}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_summary_grid(metrics_data, output_dir):
    """Create a summary grid with all important metrics using normalization."""
    plt.figure(figsize=(16, 12))
    
    # Define key metrics to highlight
    key_metrics = ['fid', 'kid_mean', 'lpips_mean', 'psnr_mean', 'ms_ssim_mean']
    
    # Create a grid
    gs = GridSpec(2, 3, figure=plt.gcf())
    
    # Create bar charts for key metrics
    for i, metric in enumerate(key_metrics):
        # Adjust placement to avoid overlap
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
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
        ax.tick_params(axis='x', labelsize=8)
    
    # Add an radar chart in the last position
    ax = plt.subplot(gs[1, 1:3], polar=True)
    
    # Use comparative normalization for the summary radar chart
    metrics_to_plot = key_metrics
    N = len(metrics_to_plot)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
    
    # Prepare normalization for all metrics using comparative method
    metric_normalized = {}
    for metric in metrics_to_plot:
        values = [data[metric] for data in metrics_data.values() if metric in data]
        if values:
            metric_normalized[metric] = normalize_radar_values(values, metric, 'comparative')
        else:
            metric_normalized[metric] = [0.6] * len(metrics_data)
    
    for i, (model, metrics) in enumerate(metrics_data.items()):
        values = []
        for j, metric in enumerate(metrics_to_plot):
            if metric in metrics:
                model_indices = [k for k, (m, _) in enumerate(metrics_data.items()) if m == model]
                if model_indices and metric in metric_normalized:
                    values.append(metric_normalized[metric][model_indices[0]])
                else:
                    values.append(0.6)
            else:
                values.append(0.6)
                
        values += values[:1]
        ax.plot(angles, values, linewidth=3, linestyle='solid', label=model, color=colors[i], marker='o', markersize=4)
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_to_plot, fontsize=10)
    ax.set_yticks([0.3, 0.5, 0.7, 0.9])
    ax.set_yticklabels(['0.3', '0.5', '0.7', '0.9'], fontsize=8)
    ax.set_ylim(0.2, 1.0)
    ax.grid(True, alpha=0.3)
    
    # Add legend with adjusted positioning
    ncol = min(len(metrics_data), 3)
    plt.figlegend(loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=ncol, fontsize=9)
    
    plt.suptitle('Model Performance Summary (Normalization)', fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'metrics_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize model evaluation metrics with normalization")
    
    # Support both single directory (backward compatibility) and multiple directories
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--metrics-dir", type=str, help="Single directory containing model metrics (backward compatibility)")
    group.add_argument("--metrics-dirs", type=str, nargs='+', help="Multiple directories containing model metrics")
    
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save visualizations")
    parser.add_argument("--normalization", type=str, default='comparative', 
                        choices=['comparative', 'performance_scaled', 'centered', 'theoretical', 'adaptive'],
                        help="Normalization method for radar charts")
    parser.add_argument("--select-models", type=str, nargs='*', default=None,
                        help="Select specific model names to include in comparison (default: include all models)")
    
    args = parser.parse_args()
    
    # Determine which argument was used
    if args.metrics_dir:
        metrics_dirs = [args.metrics_dir]
        print(f"Using single metrics directory: {args.metrics_dir}")
    else:
        metrics_dirs = args.metrics_dirs
        print(f"Using multiple metrics directories: {metrics_dirs}")
    
    # Load metrics data
    metrics_data = load_metrics(metrics_dirs, selected_models=args.select_models)
    
    if not metrics_data:
        print("No metrics data found!")
        return
    
    print(f"\nLoaded metrics for {len(metrics_data)} models:")
    for model_name in metrics_data.keys():
        print(f"  - {model_name}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    create_bar_charts(metrics_data, args.output_dir)
    create_radar_chart(metrics_data, args.output_dir, args.normalization)
    create_summary_grid(metrics_data, args.output_dir)
    
    print(f"Visualizations saved to {args.output_dir}")
    print(f"Radar charts created with normalization method: {args.normalization}")
    print("\nNormalization methods explained:")
    print("- 'comparative': Maps values to 0.3-1.0 range to avoid center clustering")
    print("- 'performance_scaled': Emphasizes relative performance differences")
    print("- 'centered': Centers around mean with reasonable spread")
    print("- 'theoretical': Uses metric-specific theoretical bounds")
    print("- 'adaptive': Automatically chooses best method based on data distribution")

if __name__ == "__main__":
    main()
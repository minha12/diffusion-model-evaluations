#!/usr/bin/env python3
# evaluation/evaluate_models.py

import os
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# TorchMetrics imports
import torchmetrics
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore

from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure,
    UniversalImageQualityIndex,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset for loading images for evaluation."""
    
    def __init__(self, image_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.image_paths = list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path.name

class DiffusionModelEvaluator:
    """Class for evaluating diffusion models using multiple metrics."""
    
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Transform for pairwise metrics (normalized float tensors)
        self.transform = T.Compose([
            T.Resize(self.config['image_size']),
            T.ToTensor(),
        ])
        
        # Transform for distribution metrics (uint8 tensors)
        self.dist_transform = T.Compose([
            T.Resize(self.config['image_size']),
            T.PILToTensor(),  # Preserves uint8 format
        ])
        
        self.metrics = self._initialize_metrics()
        
    def _initialize_metrics(self):
        """Initialize all metrics from torchmetrics."""
        metrics = {}
        
        # FID - Frechet Inception Distance
        if 'fid' in self.config['metrics']:
            metrics['fid'] = FrechetInceptionDistance(
                feature=self.config['metrics']['fid']['feature'],
                normalize=True
            ).to(self.device)
        
        # LPIPS - Learned Perceptual Image Patch Similarity
        if 'lpips' in self.config['metrics']:
            metrics['lpips'] = LearnedPerceptualImagePatchSimilarity(
                net_type=self.config['metrics']['lpips']['net_type']
            ).to(self.device)
        
        # PSNR - Peak Signal-to-Noise Ratio
        if 'psnr' in self.config['metrics']:
            metrics['psnr'] = PeakSignalNoiseRatio().to(self.device)
        
        # SSIM - Structural Similarity Index Measure
        if 'ssim' in self.config['metrics']:
            metrics['ssim'] = StructuralSimilarityIndexMeasure().to(self.device)
        
        # MS-SSIM - Multi-Scale Structural Similarity Index Measure
        if 'ms_ssim' in self.config['metrics']:
            metrics['ms_ssim'] = MultiScaleStructuralSimilarityIndexMeasure().to(self.device)
        
        # KID - Kernel Inception Distance
        if 'kid' in self.config['metrics']:
            metrics['kid'] = KernelInceptionDistance(
                subset_size=self.config['metrics']['kid']['subset_size']
            ).to(self.device)
        
        # IS - Inception Score
        if 'is' in self.config['metrics']:
            metrics['is'] = InceptionScore().to(self.device)
        
        # UIQI - Universal Image Quality Index
        if 'uiqi' in self.config['metrics']:
            metrics['uiqi'] = UniversalImageQualityIndex().to(self.device)
        
        return metrics
    
    def evaluate_model(self, generated_images_dir, ground_truth_dir, output_dir):
        """Evaluate a model by comparing generated images to ground truth."""
        logger.info(f"Evaluating images in {generated_images_dir} against {ground_truth_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load datasets with appropriate transforms
        # For distribution-based metrics (FID, KID, IS)
        generated_dist_dataset = ImageDataset(generated_images_dir, self.dist_transform)
        ground_truth_dist_dataset = ImageDataset(ground_truth_dir, self.dist_transform)
        
        # For pairwise metrics (LPIPS, PSNR, SSIM, etc.)
        generated_dataset = ImageDataset(generated_images_dir, self.transform)
        ground_truth_dataset = ImageDataset(ground_truth_dir, self.transform)
        
        # Create DataLoaders
        gen_dist_loader = DataLoader(generated_dist_dataset, batch_size=self.config['batch_size'], shuffle=False)
        gt_dist_loader = DataLoader(ground_truth_dist_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        gen_loader = DataLoader(generated_dataset, batch_size=self.config['batch_size'], shuffle=False)
        gt_loader = DataLoader(ground_truth_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Compute FID, KID, IS which need all images (using uint8 tensors)
        if 'fid' in self.metrics or 'kid' in self.metrics or 'is' in self.metrics:
            logger.info("Computing distribution-based metrics (FID, KID, IS)...")
            
            # Add real images to distribution metrics
            for real_imgs, _ in tqdm(gt_dist_loader, desc="Processing real images"):
                real_imgs = real_imgs.to(self.device)
                
                if 'fid' in self.metrics:
                    self.metrics['fid'].update(real_imgs, real=True)
                
                if 'kid' in self.metrics:
                    self.metrics['kid'].update(real_imgs, real=True)
                
                if 'is' in self.metrics:
                    self.metrics['is'].update(real_imgs)
            
            # Add generated images to distribution metrics
            for gen_imgs, _ in tqdm(gen_dist_loader, desc="Processing generated images"):
                gen_imgs = gen_imgs.to(self.device)
                
                if 'fid' in self.metrics:
                    self.metrics['fid'].update(gen_imgs, real=False)
                
                if 'kid' in self.metrics:
                    self.metrics['kid'].update(gen_imgs, real=False)
        
        # Compute pairwise metrics (LPIPS, PSNR, SSIM, MS-SSIM, UIQI)
        pairwise_results = {
            'lpips': [],
            'psnr': [],
            'ssim': [],
            'ms_ssim': [],
            'uiqi': []
        }
        
        logger.info("Computing pairwise metrics...")
        for (gen_imgs, gen_names), (real_imgs, real_names) in tqdm(
            zip(gen_loader, gt_loader), total=min(len(gen_loader), len(gt_loader)), 
            desc="Computing pairwise metrics"
        ):
            gen_imgs = gen_imgs.to(self.device)
            real_imgs = real_imgs.to(self.device)
            
            # Only compute pairwise metrics for matching image pairs
            for metric_name in ['lpips', 'psnr', 'ssim', 'ms_ssim', 'uiqi']:
                if metric_name in self.metrics:
                    result = self.metrics[metric_name](gen_imgs, real_imgs)
                    # Make sure the result is at least 1D
                    result_np = result.cpu().numpy()
                    # Reshape scalar results to 1D arrays if needed
                    if result_np.ndim == 0:
                        result_np = result_np.reshape(1)
                    pairwise_results[metric_name].append(result_np)
        
        # Compute final results
        results = {}
        
        # Distribution-based metrics
        if 'fid' in self.metrics:
            results['fid'] = self.metrics['fid'].compute().item()
            logger.info(f"FID: {results['fid']:.4f}")
        
        if 'kid' in self.metrics:
            kid_mean, kid_std = self.metrics['kid'].compute()
            results['kid_mean'] = kid_mean.item()
            results['kid_std'] = kid_std.item()
            logger.info(f"KID: {results['kid_mean']:.4f} ± {results['kid_std']:.4f}")
        
        if 'is' in self.metrics:
            is_mean, is_std = self.metrics['is'].compute()
            results['is_mean'] = is_mean.item()
            results['is_std'] = is_std.item()
            logger.info(f"IS: {results['is_mean']:.4f} ± {results['is_std']:.4f}")
        
        # Pairwise metrics
        for metric_name in ['lpips', 'psnr', 'ssim', 'ms_ssim', 'uiqi']:
            if metric_name in self.metrics and pairwise_results[metric_name]:
                # Only proceed if we have values to concatenate
                if len(pairwise_results[metric_name]) > 0:
                    values = np.concatenate(pairwise_results[metric_name])
                    results[f'{metric_name}_mean'] = float(np.mean(values))
                    results[f'{metric_name}_std'] = float(np.std(values))
                    logger.info(f"{metric_name.upper()}: {results[f'{metric_name}_mean']:.4f} ± {results[f'{metric_name}_std']:.4f}")
        
        # Save results
        output_file = os.path.join(output_dir, 'metrics_results.yaml')
        with open(output_file, 'w') as f:
            yaml.dump(results, f)
        
        logger.info(f"Results saved to {output_file}")
        
        # Generate visualizations
        self._generate_visualizations(results, output_dir)
        
        return results
    
    def _generate_visualizations(self, results, output_dir):
        """Generate visualizations of the evaluation results."""
        # Create a bar chart for the metrics
        plt.figure(figsize=(12, 6))
        
        # Filter metrics for visualization (use means for metrics with std)
        vis_metrics = {k: v for k, v in results.items() if not k.endswith('_std')}
        
        # Sort metrics by name
        sorted_metrics = sorted(vis_metrics.items())
        
        # Create the bar chart
        plt.bar(range(len(sorted_metrics)), [val for _, val in sorted_metrics])
        plt.xticks(range(len(sorted_metrics)), [name for name, _ in sorted_metrics], rotation=45)
        plt.title('Evaluation Metrics')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, 'metrics_visualization.png'))
        plt.close()
        
        logger.info(f"Visualization saved to {os.path.join(output_dir, 'metrics_visualization.png')}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate diffusion models using multiple metrics")
    parser.add_argument("--config", type=str, required=True, help="Path to the evaluation config file")
    parser.add_argument("--generated", type=str, required=True, help="Directory containing generated images")
    parser.add_argument("--ground-truth", type=str, required=True, help="Directory containing ground truth images")
    parser.add_argument("--output", type=str, required=True, help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    evaluator = DiffusionModelEvaluator(args.config)
    evaluator.evaluate_model(args.generated, args.ground_truth, args.output)
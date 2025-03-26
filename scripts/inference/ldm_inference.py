import torch
import numpy as np
import os
from pathlib import Path
import sys
import fire
import glob

# Add the latent-diffusion-semantic repo to the Python path
def add_ldm_to_path():
    ldm_path = Path("models/latent-diffusion-semantic")
    if ldm_path.exists() and ldm_path not in sys.path:
        sys.path.append(str(ldm_path))

add_ldm_to_path()

# Now we can import from the LDM repository
from scripts.sample_diffusion import load_model
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from einops import rearrange
from PIL import Image
import albumentations
import cv2

# Custom dataset that only needs segmentation files
class SegmentationOnlyDataset(Dataset):
    def __init__(self, segmentation_dir, size=512, n_labels=5):
        """
        Dataset that only requires segmentation masks for LDM inference.
        
        Args:
            segmentation_dir: Directory containing segmentation PNG files
            size: Size to resize images to (default: 512)
            n_labels: Number of segmentation labels (default: 5)
        """
        self.segmentation_dir = segmentation_dir
        self.size = size
        self.n_labels = n_labels
        
        # Find all PNG files in the directory
        self.segmentation_paths = sorted(glob.glob(os.path.join(segmentation_dir, "*.png")))
        self._length = len(self.segmentation_paths)
        
        if self._length == 0:
            raise ValueError(f"No PNG files found in {segmentation_dir}")
            
        print(f"Found {self._length} segmentation files in {segmentation_dir}")
        
        # Setup resizing transformations
        self.segmentation_rescaler = albumentations.Resize(
            height=self.size, 
            width=self.size,
            interpolation=cv2.INTER_NEAREST
        )

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # Get the file path
        seg_path = self.segmentation_paths[i]
        file_name = os.path.basename(seg_path).replace('.png', '')
        
        # Load segmentation
        segmentation = Image.open(seg_path)
        assert segmentation.mode == "L", f"Expected grayscale image, got {segmentation.mode}"
        segmentation = np.array(segmentation).astype(np.uint8)
        
        # Resize if needed
        if self.size is not None:
            segmentation = self.segmentation_rescaler(image=segmentation)["image"]
            
        # Create one-hot encoding
        onehot = np.eye(self.n_labels)[segmentation]
        
        # Create example dictionary
        example = {
            "segmentation": onehot,
            "file_names": file_name
        }
        
        return example


def ldm_cond_sample(config_path, ckpt_path, dataset, batch_size, steps, save_dir, seed=42):
    """Generate samples from a conditional latent diffusion model"""
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    
    # Load config and model
    config = OmegaConf.load(config_path)
    model, _ = load_model(config, ckpt_path, None, None)
    
    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process each batch
    for i, batch in enumerate(dataloader):
        seg = batch['segmentation']
        
        # Get file names if available in the dataset
        file_names = batch.get('file_names', [f"sample_{i*batch_size + j}" for j in range(len(seg))])
        
        print(f"Processing batch {i}, shape: {seg.shape}")
        
        with torch.no_grad():
            # Rearrange segmentation from [B, H, W, C] to [B, C, H, W]
            seg = rearrange(seg, 'b h w c -> b c h w')
            
            # Visualize condition
            if hasattr(model, 'to_rgb'):
                condition = model.to_rgb(seg)
                # Condition visualization removed
            
            # Move to GPU and convert to float
            seg = seg.to('cuda').float()
            
            # Get learned conditioning
            cond = model.get_learned_conditioning(seg)
            
            # Sample from the model
            samples, _ = model.sample_log(
                cond=cond, 
                batch_size=len(seg), 
                ddim=True,
                ddim_steps=steps, 
                eta=1.0
            )
            
            # Decode the samples
            samples = model.decode_first_stage(samples)
        
        # Save generated samples
        for j, sample in enumerate(samples):
            save_image(sample, os.path.join(save_dir, f"{file_names[j]}.png"))


def main(
    control_images, 
    output_dir, 
    config_path, 
    ckpt_path, 
    batch_size=4, 
    steps=200, 
    seed=42, 
    size=512,  # Added argument
    n_labels=5  # Added argument
):
    """
    Run inference using LDM model.
    
    Args:
        control_images: List of paths to control images
        output_dir: Output directory for generated images
        config_path: Path to LDM config YAML
        ckpt_path: Path to LDM checkpoint
        batch_size: Batch size for inference (default: 4)
        steps: Number of diffusion steps (default: 200)
        seed: Random seed (default: 42)
        size: Size to resize images to (default: 512)
        n_labels: Number of segmentation labels (default: 5)
    """
    # Set random seed
    torch.manual_seed(seed)
    
    # Create dataset from provided control images
    dataset = SegmentationOnlyDataset(
        segmentation_dir=control_images,
        size=size,  # Pass size argument
        n_labels=n_labels  # Pass n_labels argument
    )
    
    # Run the sampling function
    ldm_cond_sample(
        config_path=config_path,
        ckpt_path=ckpt_path,
        dataset=dataset,
        batch_size=batch_size,
        steps=steps,
        save_dir=output_dir,
        seed=seed
    )


if __name__ == '__main__':
    fire.Fire(main)
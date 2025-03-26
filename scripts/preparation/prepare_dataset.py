#!/usr/bin/env python3

import os
import shutil
import json
import random
from pathlib import Path
import fire
from tqdm import tqdm  # Add this import

def ensure_dir_exists(dir_path):
    os.makedirs(dir_path, exist_ok=True)

def create_directory_structure(target_dir):
    """Create the directory structure for the test dataset"""
    dirs = [
        "patches",
        "prompts",
        "segmentation",
        "plain-segmentation"  # Added new directory
    ]
    
    for dir_path in dirs:
        ensure_dir_exists(os.path.join(target_dir, dir_path))

def read_prompt_json(source_dir, prompt_file="prompt.json"):
    """Read the prompt file
    
    Args:
        source_dir: Source directory containing the dataset
        prompt_file: Name of the prompt file (default: "prompt.json")
    """
    prompt_path = os.path.join(source_dir, prompt_file)
    with open(prompt_path, 'r') as f:
        # Read the file line by line as each line is a JSON object
        prompts = [json.loads(line) for line in f]
    return prompts

def extract_image_id(path):
    """Extract the image ID from a file path"""
    return os.path.splitext(os.path.basename(path))[0]

def match_prompt_to_image(prompts, image_id):
    """Find the prompt that matches the image ID"""
    for prompt_data in prompts:
        source_id = extract_image_id(prompt_data['source'])
        target_id = extract_image_id(prompt_data['target'])
        
        # Check if either source or target match the image_id
        if source_id == image_id or target_id == image_id:
            return prompt_data['prompt']
    
    return None

def copy_dataset(
    source_dir="~/datasets/drsk", 
    target_dir="~/diffusion_models_evaluation/data/new_test_dataset", 
    num_samples=5, 
    seed=42,
    images_subdir="images",
    segmentation_subdir="segmentation",
    plain_segmentation_subdir="plain-segmentation",  # Added new parameter
    prompt_file="prompt.json",
    verbose=False
):
    """
    Copy files from source to target directory to create a test dataset.
    
    Args:
        source_dir: Source directory containing the drsk dataset
        target_dir: Target directory to create the new test dataset
        num_samples: Number of samples to copy
        seed: Random seed for reproducibility
        images_subdir: Subdirectory name containing images (default: "images")
        segmentation_subdir: Subdirectory name containing segmentation masks (default: "segmentation")
        plain_segmentation_subdir: Subdirectory name containing plain segmentation masks (default: "plain-segmentation")
        prompt_file: Name of the prompt file (default: "prompt.json")
        verbose: Whether to print detailed information (default: False)
    """
    source_dir = os.path.expanduser(source_dir)
    target_dir = os.path.expanduser(target_dir)
    
    print(f"Preparing dataset: copying {num_samples} samples from {source_dir} to {target_dir}")
    
    # Create the directory structure
    create_directory_structure(target_dir)
    
    # Read prompts
    prompts = read_prompt_json(source_dir, prompt_file)
    
    # Get list of image files
    images_dir = os.path.join(source_dir, images_subdir)
    segmentation_dir = os.path.join(source_dir, segmentation_subdir)
    plain_segmentation_dir = os.path.join(source_dir, plain_segmentation_subdir)  # Added new directory reference
    
    # List all images and get their IDs
    all_image_files = os.listdir(images_dir)
    image_ids = [extract_image_id(img) for img in all_image_files]
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Randomly select image IDs
    selected_ids = random.sample(image_ids, min(num_samples, len(image_ids)))
    
    # Create prompts.txt file
    prompts_file = os.path.join(target_dir, "prompts", "prompts.txt")
    
    # Add progress bar
    pbar = tqdm(selected_ids, desc="Copying dataset files", unit="sample")
    
    with open(prompts_file, 'w') as f:
        for i, image_id in enumerate(pbar):
            if verbose:
                pbar.set_description(f"Processing {image_id}")
                
            # Copy image to patches folder
            src_image = os.path.join(images_dir, f"{image_id}.jpg")
            if not os.path.exists(src_image):
                src_image = os.path.join(images_dir, f"{image_id}.png")
            
            # Use original filename instead of sequential numbering
            image_filename = f"{image_id}{os.path.splitext(src_image)[1]}"
            target_image = os.path.join(target_dir, "patches", image_filename)
            
            if os.path.exists(src_image):
                shutil.copy2(src_image, target_image)
                if verbose:
                    print(f"Copied image: {src_image} -> {target_image}")
            else:
                if verbose:
                    print(f"Warning: Source image not found: {src_image}")
            
            # Copy segmentation mask
            src_seg = os.path.join(segmentation_dir, f"{image_id}.png")
            target_seg = os.path.join(target_dir, "segmentation", f"{image_id}.png")
            
            if os.path.exists(src_seg):
                shutil.copy2(src_seg, target_seg)
                if verbose:
                    print(f"Copied segmentation: {src_seg} -> {target_seg}")
            else:
                if verbose:
                    print(f"Warning: Source segmentation not found: {src_seg}")
            
            # After copying segmentation, add code to copy plain-segmentation
            src_plain_seg = os.path.join(plain_segmentation_dir, f"{image_id}.png")
            target_plain_seg = os.path.join(target_dir, "plain-segmentation", f"{image_id}.png")
            
            if os.path.exists(src_plain_seg):
                shutil.copy2(src_plain_seg, target_plain_seg)
                if verbose:
                    print(f"Copied plain segmentation: {src_plain_seg} -> {target_plain_seg}")
            else:
                if verbose:
                    print(f"Warning: Source plain segmentation not found: {src_plain_seg}")
            
            # Write prompt to file
            prompt_text = match_prompt_to_image(prompts, image_id)
            if prompt_text:
                f.write(f"{prompt_text}\n")
                if verbose:
                    print(f"Added prompt for image {image_id}: {prompt_text}")
            else:
                # If no prompt is found, create a generic one based on the pattern in your example
                generic_prompt = "pathology image: tissue unknown 50.00%, dermis normal skin 50.00%"
                f.write(f"{generic_prompt}\n")
                if verbose:
                    print(f"No prompt found for image {image_id}, using generic prompt")
    
    print(f"\nDataset successfully copied to {target_dir}")
    print(f"Created {len(selected_ids)} samples with corresponding images, segmentations, and prompts")

if __name__ == "__main__":
    fire.Fire(copy_dataset)
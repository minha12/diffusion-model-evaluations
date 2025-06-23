#!/usr/bin/env python3
import os
import argparse
import random
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import yaml # To read the main config

def get_image_files(data_dir):
    """Gets a list of image file paths from a directory."""
    image_extensions = ['.png', '.jpg', '.jpeg']
    return [p for p in Path(data_dir).glob('*') if p.suffix.lower() in image_extensions]

def derive_classification_label(mask_path, strategy, carcinoma_threshold, num_classes_seg):
    """Derives a classification label from a segmentation mask."""
    try:
        mask = Image.open(mask_path)
        mask_np = np.array(mask)

        # Check if mask is grayscale (H, W) or RGB (H, W, C)
        if mask_np.ndim == 3:
            # If RGB, need a mapping or assume grayscale conversion logic
            # Assuming it's indexed PNG already (plain-segmentation)
             logger.warning(f"Mask {mask_path} appears to be RGB. Assuming index is in the first channel.")
             mask_np = mask_np[:,:,0] # Example: Adapt if needed

        if mask_np.size == 0:
             return -1 # Invalid mask

        counts = np.bincount(mask_np.flatten(), minlength=num_classes_seg)
        total_pixels = mask_np.size
        if total_pixels == 0:
             return -1 # Avoid division by zero

        # --- Binary Carcinoma Strategy ---
        if strategy == "binary_carcinoma":
            carcinoma_pixels = counts[3] # Class ID 3 is Carcinoma
            if carcinoma_pixels / total_pixels >= carcinoma_threshold:
                return 1  # Carcinoma present
            else:
                return 0  # Other

        # --- Multiclass Dominant Strategy ---
        elif strategy == "multiclass_dominant":
            # Exclude Unknown (0) and Background (1) for dominant tissue type
            valid_counts = counts[2:] # Index 2, 3, 4
            if np.sum(valid_counts) == 0:
                # If only Unknown/Background, assign a default label (e.g., Normal or a specific 'Other')
                 # Let's map it to the 'Normal' index relative to the reduced class set (Carcinoma=1, Normal=2)
                 # Find the index corresponding to 'Normal' in the reduced class list
                 # For ["Inflammatory/Reactive", "Carcinoma", "Normal"], Normal is index 2.
                 # Return the index relative to the start of valid_counts (index 2 in original)
                 return 2 # Mapping to the index for 'Normal' in the reduced class list

            dominant_class_relative = np.argmax(valid_counts)
            dominant_class_absolute = dominant_class_relative + 2 # Add back the offset
            # Map absolute class (2, 3, 4) to the output classes (0, 1, 2 for the example multiclass names)
            # Inflammatory/Reactive (2) -> 0
            # Carcinoma (3) -> 1
            # Normal (4) -> 2
            mapping = {2: 0, 3: 1, 4: 2}
            return mapping.get(dominant_class_absolute, -1) # Return mapped class or -1 if error

        else:
            raise ValueError(f"Unknown labeling strategy: {strategy}")

    except Exception as e:
        print(f"Error processing mask {mask_path}: {e}")
        return -1 # Indicate error

def main(config_path, base_data_dir, output_dir, test_split_ratio=0.2):
    """
    Prepares metadata files for downstream classification and segmentation tasks.

    Args:
        config_path: Path to the main evaluation config YAML.
        base_data_dir: Path to the specific dataset directory (e.g., data/drsk_1k_seed_42).
        output_dir: Directory to save the metadata files.
        test_split_ratio: Fraction of real data to use for the test set.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    ds_config = config['downstream_tasks']
    cls_config = ds_config.get('classification', {})
    seg_config = ds_config.get('segmentation', {})
    num_seg_classes = seg_config.get('num_classes', 5) # Get number of segmentation classes

    real_patches_dir = Path(base_data_dir) / "patches"
    real_masks_dir = Path(base_data_dir) / "plain-segmentation" # Use plain masks
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Looking for real images in: {real_patches_dir}")
    print(f"Looking for real masks in: {real_masks_dir}")

    real_image_paths = get_image_files(real_patches_dir)
    if not real_image_paths:
        raise FileNotFoundError(f"No real images found in {real_patches_dir}")

    data = []
    print("Processing real images and masks...")
    for img_path in tqdm(real_image_paths, desc="Processing real data"):
        img_id = img_path.stem
        mask_path = real_masks_dir / f"{img_id}.png"

        if not mask_path.exists():
            print(f"Warning: Mask not found for image {img_id}. Skipping.")
            continue

        label = -1 # Default invalid label
        if cls_config.get('enabled', False):
            label = derive_classification_label(
                mask_path,
                cls_config.get('labeling_strategy', 'binary_carcinoma'),
                cls_config.get('carcinoma_threshold', 0.1),
                num_seg_classes # Pass the number of segmentation classes
            )
            if label == -1:
                 print(f"Warning: Could not derive label for image {img_id}. Skipping.")
                 continue # Skip if label derivation failed

        data.append({
            "image_id": img_id,
            "real_image_path": str(img_path.relative_to(Path(config['base_dir'])) if 'base_dir' in config else img_path), # Store relative paths if base_dir is known
            "real_mask_path": str(mask_path.relative_to(Path(config['base_dir'])) if 'base_dir' in config else mask_path),
            "classification_label": label
        })

    if not data:
        raise ValueError("No valid image/mask pairs found or processed.")

    df = pd.DataFrame(data)

    # Split real data into train/test
    unique_ids = df["image_id"].unique()
    random.shuffle(unique_ids)
    test_size = int(len(unique_ids) * test_split_ratio)
    test_ids = set(unique_ids[:test_size])
    train_ids = set(unique_ids[test_size:])

    df["split"] = df["image_id"].apply(lambda x: "test" if x in test_ids else "train")

    # Save the metadata
    metadata_path = output_dir / "downstream_metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Downstream task metadata saved to: {metadata_path}")
    print(f"Real data split: {len(train_ids)} train, {len(test_ids)} test.")

    # Create separate files for convenience if needed (optional)
    df[df["split"] == "train"].to_csv(output_dir / "train_metadata.csv", index=False)
    df[df["split"] == "test"].to_csv(output_dir / "test_metadata.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare metadata for downstream tasks.")
    parser.add_argument("--config", required=True, help="Path to the main evaluation config YAML.")
    parser.add_argument("--dataset-dir", required=True, help="Path to the specific dataset directory (e.g., data/drsk_1k_seed_42).")
    parser.add_argument("--output-dir", required=True, help="Directory to save metadata files.")
    parser.add_argument("--test-split", type=float, default=0.2, help="Fraction of real data for the test set.")
    # Add base_dir argument to help make paths relative
    parser.add_argument("--base-dir", default=".", help="Project base directory for creating relative paths.")


    args = parser.parse_args()

    # Add base_dir to config temporarily for path relativization in the function
    temp_config_path = args.config
    with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    config_data['base_dir'] = args.base_dir # Inject base_dir

    # Save temporary config if needed, or pass base_dir differently
    # For simplicity here, we assume the main function can access it via the loaded config dict
    # If not, pass args.base_dir directly to main()

    main(args.config, args.dataset_dir, args.output_dir, args.test_split)
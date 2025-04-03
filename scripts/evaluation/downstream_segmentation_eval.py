#!/usr/bin/env python3
import os
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import torchmetrics # Use torchmetrics

# --- Import segmentation_models_pytorch ---
import segmentation_models_pytorch as smp

# --- Configuration ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, metadata_df, image_dir, mask_dir, img_transform=None, mask_transform=None, use_generated=False, generated_suffix=".png", base_dir="."):
        self.metadata = metadata_df
        self.image_dir = Path(base_dir) / image_dir if image_dir else None # Allow empty image_dir for real data paths in metadata
        self.mask_dir = Path(base_dir) / mask_dir if mask_dir else None # Masks always come from real data path in metadata
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.use_generated = use_generated
        self.generated_suffix = generated_suffix
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']

        # --- Get Image Path ---
        if self.use_generated and self.image_dir:
            img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
        else:
            img_path_relative = row['real_image_path']
            img_path = self.base_dir / img_path_relative

        # --- Get Mask Path ---
        mask_path_relative = row['real_mask_path'] # Always use real mask path from metadata
        mask_path = self.base_dir / mask_path_relative

        if not img_path.exists():
             print(f"Warning: Image not found at {img_path}. Skipping.")
             return None, None
        if not mask_path.exists():
             print(f"Warning: Mask not found at {mask_path}. Skipping.")
             return None, None

        try:
            image = Image.open(img_path).convert('RGB')
            # Use plain-segmentation masks (assume single channel, 0-4 values)
            mask = Image.open(mask_path) # Should be L mode (grayscale)

            # Apply transforms
            if self.img_transform:
                image = self.img_transform(image)

            # Convert mask to tensor *without* normalization
            # Mask should contain class indices (long type)
            if self.mask_transform:
                 mask = self.mask_transform(mask) # Apply geometric transforms like resize/crop
            else:
                 # Basic conversion if no other transform
                 img_size = tuple(load_config('config/evaluation.yaml').get('image_size', [256, 256])) # Get size
                 mask = T.functional.resize(mask, img_size, interpolation=T.InterpolationMode.NEAREST) # Ensure size match, use NEAREST for masks

            mask = torch.from_numpy(np.array(mask)).long() # Convert to LongTensor

            # Remove channel dimension if mask is (1, H, W) -> (H, W)
            if mask.ndim == 3 and mask.shape[0] == 1:
                 mask = mask.squeeze(0)

        except Exception as e:
            print(f"Error loading image/mask pair {image_id}: {e}")
            return None, None

        return image, mask

# --- Model ---
def get_segmentation_model(model_name, encoder, num_classes, pretrained='imagenet'):
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
        )
        print(f"Using segmentation_models_pytorch U-Net with encoder: {encoder}")
    else:
        raise ValueError(f"Segmentation model {model_name} not supported.")
    return model

# --- Training ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
         # Filter out None items caused by loading errors
        valid_indices = [i for i, (img, msk) in enumerate(zip(images, masks)) if img is not None]
        if not valid_indices:
            continue

        images = torch.stack([images[i] for i in valid_indices]).to(device)
        masks = torch.stack([masks[i] for i in valid_indices]).to(device) # Masks should be LongTensor (H, W)

        optimizer.zero_grad()
        outputs = model(images) # Shape: (B, C, H, W)
        loss = criterion(outputs, masks) # CrossEntropyLoss expects (B, C, H, W) and (B, H, W)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset) # Adjust divisor if dataset contains None
    return epoch_loss

# --- Evaluation ---
def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0

    # Use torchmetrics for mIoU and Pixel Accuracy
    # ignore_index=0 might be useful if 'Unknown' class should be excluded from metric
    miou_metric = torchmetrics.JaccardIndex(task="multiclass", num_classes=num_classes, average='macro').to(device) # mIoU
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device) # Pixel Accuracy

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            # Filter out None items
            valid_indices = [i for i, (img, msk) in enumerate(zip(images, masks)) if img is not None]
            if not valid_indices:
                 continue

            images = torch.stack([images[i] for i in valid_indices]).to(device)
            masks = torch.stack([masks[i] for i in valid_indices]).to(device)

            outputs = model(images) # (B, C, H, W)
            loss = criterion(outputs, masks) # (B, H, W)
            running_loss += loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1) # (B, H, W)

            # Update metrics
            miou_metric.update(preds, masks)
            accuracy_metric.update(preds, masks)

    eval_loss = running_loss / len(dataloader.dataset) # Adjust divisor if dataset contains None

    # Compute final metrics
    final_miou = miou_metric.compute().item()
    final_accuracy = accuracy_metric.compute().item()

    metrics = {
        "loss": eval_loss,
        "mean_iou": final_miou,
        "pixel_accuracy": final_accuracy,
    }

    # Reset metrics
    miou_metric.reset()
    accuracy_metric.reset()

    return metrics

# --- Main Function ---
def main(args):
    config = load_config(args.config_path)
    seg_config = config['downstream_tasks']['segmentation']
    num_classes = seg_config['num_classes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    metadata_df = pd.read_csv(args.metadata_path)
    base_dir = Path(args.base_dir)

    if args.mode == "baseline":
        train_df = metadata_df[metadata_df['split'] == 'train']
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = "" # Use paths from metadata
        test_image_dir = ""  # Use paths from metadata
        use_train_generated = False
        print(f"Running Baseline: Training on {len(train_df)} real samples, Testing on {len(test_df)} real samples.")
    elif args.mode == "generated":
        train_df = metadata_df # Use all real samples for training mapping
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = args.generated_images_dir # Relative path
        test_image_dir = "" # Use paths from metadata
        use_train_generated = True
        print(f"Running Generated Eval: Training on {len(train_df)} generated samples, Testing on {len(test_df)} real samples.")
    else:
        raise ValueError("Mode must be 'baseline' or 'generated'")

    # Define transforms
    img_size = tuple(config.get('image_size', [256, 256]))
    img_transform = T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # Mask transform only needs resizing (nearest neighbor)
    mask_transform = T.Compose([
        T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST),
    ])

    # Create datasets
    train_dataset = SegmentationDataset(train_df, train_image_dir, mask_dir="", img_transform=img_transform, mask_transform=mask_transform, use_generated=use_train_generated, base_dir=args.base_dir)
    test_dataset = SegmentationDataset(test_df, test_image_dir, mask_dir="", img_transform=img_transform, mask_transform=mask_transform, use_generated=False, base_dir=args.base_dir) # Always test on real

    # Collate function to handle loading errors
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and x[0] is not None and x[1] is not None, batch))
        if not batch: return torch.Tensor(), torch.Tensor()
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(train_dataset, batch_size=seg_config['batch_size'], shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=seg_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Model Initialization ---
    model = get_segmentation_model(
        seg_config['model'],
        seg_config.get('encoder', 'resnet34'), # Provide default encoder
        num_classes
    ).to(device)
    # Use CrossEntropyLoss for multi-class segmentation
    criterion = nn.CrossEntropyLoss() # Can add class weights here if needed
    optimizer = optim.Adam(model.parameters(), lr=seg_config['learning_rate'])

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(seg_config['num_epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{seg_config['num_epochs']} - Train Loss: {train_loss:.4f}")
        # Optional: Add validation loop here if desired

    # --- Final Evaluation ---
    print("Starting final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print("-" * 30)
    print("Test Set Evaluation Results:")
    for key, value in test_metrics.items():
        print(f"{key.replace('_', ' ').capitalize()}: {value:.4f}")
    print("-" * 30)

    # --- Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "segmentation_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Segmentation results saved to {results_path}")

    # Optionally save the trained model
    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{args.mode}_segmentation_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Trained model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream Segmentation Evaluation")
    parser.add_argument("--config-path", required=True, help="Path to the main evaluation config YAML.")
    parser.add_argument("--metadata-path", required=True, help="Path to the downstream metadata CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--mode", required=True, choices=["baseline", "generated"], help="Evaluation mode.")
    parser.add_argument("--generated-images-dir", help="Directory containing generated images (required if mode='generated'). Relative to base_dir.")
    parser.add_argument("--base-dir", default=".", help="Project base directory for resolving relative paths.")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model.")

    args = parser.parse_args()

    if args.mode == "generated" and not args.generated_images_dir:
        parser.error("--generated-images-dir is required when mode='generated'")

    main(args)
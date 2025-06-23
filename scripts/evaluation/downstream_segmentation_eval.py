#!/usr/bin/env python3
import os
import argparse
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import torchmetrics
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

# --- Import segmentation_models_pytorch ---
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses as smp_losses

# --- Configuration ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Enhanced Augmentation Functions ---
def get_training_augmentation(img_size, encoder_name=None):
    """Enhanced training augmentations with better medical image support"""
    if encoder_name:
        try:
            params = smp.encoders.get_preprocessing_params(encoder_name)
            mean = params.get('mean', (0.485, 0.456, 0.406))
            std = params.get('std', (0.229, 0.224, 0.225))
        except:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    
    train_transform = [
        A.Resize(height=img_size[0], width=img_size[1]),
        
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),  # Added vertical flip
        A.RandomRotate90(p=0.3),  # Added 90-degree rotations
        A.Transpose(p=0.3),  # Added transpose
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(-0.1, 0.1),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.5,
            mode=0
        ),
        
        # Intensity transformations
        A.OneOf([
            A.CLAHE(clip_limit=2.0, p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(gamma_limit=(80, 120), p=1),
        ], p=0.8),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10, 50), p=1),
            A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=1),
        ], p=0.3),
        
        A.OneOf([
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),
            A.MedianBlur(blur_limit=3, p=1),
        ], p=0.3),
        
        # Color augmentations (lighter for medical images)
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=15,
            val_shift_limit=10,
            p=0.3
        ),
        
        # Grid distortion for better boundary learning
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=1),
            A.ElasticTransform(alpha=1, sigma=50, p=1),
        ], p=0.2),
        
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(train_transform)

def get_validation_testing_augmentation(img_size, encoder_name=None):
    """Clean validation/testing augmentations"""
    if encoder_name:
        try:
            params = smp.encoders.get_preprocessing_params(encoder_name)
            mean = params.get('mean', (0.485, 0.456, 0.406))
            std = params.get('std', (0.229, 0.224, 0.225))
        except:
            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)
    else:
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        
    test_transform = [
         A.Resize(height=img_size[0], width=img_size[1]),
         A.Normalize(mean=mean, std=std),
         ToTensorV2(),
    ]
    return A.Compose(test_transform)

# --- Enhanced Dataset with better error handling ---
class SegmentationDataset(Dataset):
    def __init__(self, metadata_df, image_dir, mask_dir, augmentation=None, use_generated=False, generated_suffix=".png", base_dir="."):
        self.metadata = metadata_df
        self.image_dir = Path(base_dir) / image_dir if image_dir else None
        self.mask_dir = Path(base_dir) / mask_dir if mask_dir else None
        self.augmentation = augmentation
        self.use_generated = use_generated
        self.generated_suffix = generated_suffix
        self.base_dir = Path(base_dir)
        
        # Pre-validate dataset
        self.valid_indices = self._validate_dataset()
        print(f"Dataset initialized with {len(self.valid_indices)}/{len(self.metadata)} valid samples")

    def _validate_dataset(self):
        """Pre-validate all samples to avoid runtime errors"""
        valid_indices = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            image_id = row['image_id']
            
            # Get paths
            if self.use_generated and self.image_dir:
                img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
            else:
                img_path_relative = row['real_image_path']
                img_path = self.base_dir / img_path_relative
            
            mask_path_relative = row['real_mask_path']
            mask_path = self.base_dir / mask_path_relative
            
            if img_path.exists() and mask_path.exists():
                valid_indices.append(idx)
        
        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.metadata.iloc[actual_idx]
        image_id = row['image_id']

        # Get paths
        if self.use_generated and self.image_dir:
            img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
        else:
            img_path_relative = row['real_image_path']
            img_path = self.base_dir / img_path_relative

        mask_path_relative = row['real_mask_path']
        mask_path = self.base_dir / mask_path_relative

        try:
            # Load and process image
            image = Image.open(str(img_path)).convert('RGB')
            image = np.array(image)
            
            # Load and process mask
            mask = Image.open(str(mask_path)).convert('L')
            mask = np.array(mask)
            
            # Ensure mask values are in valid range
            mask = np.clip(mask, 0, 4)  # Assuming 5 classes (0-4)

            # Apply augmentations
            if self.augmentation:
                transformed = self.augmentation(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            # Ensure mask is LongTensor
            if isinstance(mask, torch.Tensor):
                mask = mask.long()
            else:
                mask = torch.from_numpy(mask).long()
                
            return image, mask

        except Exception as e:
            print(f"Error loading image/mask pair {image_id}: {e}")
            # Return a dummy sample to maintain batch consistency
            dummy_image = torch.zeros(3, 256, 256)
            dummy_mask = torch.zeros(256, 256, dtype=torch.long)
            return dummy_image, dummy_mask

# --- Enhanced Model with better architectures ---
def get_segmentation_model(model_name, encoder, num_classes, pretrained='imagenet'):
    """Get segmentation model with enhanced options"""
    model_name = model_name.lower()
    
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
            activation=None,  # We'll handle activation in loss
        )
    elif model_name == 'unetplusplus':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    elif model_name == 'deeplabv3plus':
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )
    else:
        raise ValueError(f"Segmentation model {model_name} not supported.")
    
    print(f"Using {model_name.upper()} with encoder: {encoder}")
    return model

# --- Enhanced Loss Function ---
class CombinedLoss(nn.Module):
    """Combined loss for better segmentation performance"""
    def __init__(self, num_classes, alpha=0.7, beta=0.3, class_weights=None):
        super().__init__()
        self.dice_loss = smp_losses.DiceLoss(mode='multiclass', from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, logits, targets):
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        return self.alpha * dice + self.beta * ce

# --- Enhanced Training with Validation ---
def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    model.train()
    running_loss = 0.0
    processed_samples = 0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item() * images.size(0)
        processed_samples += images.size(0)

    epoch_loss = running_loss / processed_samples
    return epoch_loss

def validate_epoch(model, dataloader, criterion, device, num_classes):
    """Validation during training - FIXED VERSION"""
    model.eval()
    running_loss = 0.0
    processed_samples = 0
    
    # Metrics
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Initialize metric accumulators as tensors with correct shape
    tp_sum = torch.zeros(num_classes, device=device)
    fp_sum = torch.zeros(num_classes, device=device)
    fn_sum = torch.zeros(num_classes, device=device)
    tn_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Validating", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, masks)
            
            # IoU computation - FIX: Sum across batch dimension
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, masks, mode='multiclass', num_classes=num_classes
            )
            
            # Ensure tensors are on the same device and sum across batch dimension
            tp = tp.to(device).sum(dim=0)  # Sum across batch dimension
            fp = fp.to(device).sum(dim=0)  # Sum across batch dimension
            fn = fn.to(device).sum(dim=0)  # Sum across batch dimension
            tn = tn.to(device).sum(dim=0)  # Sum across batch dimension
            
            # Accumulate with proper tensor operations
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            tn_sum += tn

    val_loss = running_loss / processed_samples
    val_accuracy = accuracy_metric.compute().item()
    val_miou = smp.metrics.iou_score(tp_sum, fp_sum, fn_sum, tn_sum, reduction="micro").item()
    
    accuracy_metric.reset()
    
    return {
        "loss": val_loss,
        "accuracy": val_accuracy,
        "miou": val_miou
    }

# --- Enhanced Evaluation ---
def evaluate(model, dataloader, criterion, device, num_classes):
    """Enhanced evaluation - FIXED VERSION"""
    model.eval()
    running_loss = 0.0
    processed_samples = 0

    # Metrics
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, average='macro').to(device)
    
    # Initialize metric accumulators as tensors with correct shape
    tp_sum = torch.zeros(num_classes, device=device)
    fp_sum = torch.zeros(num_classes, device=device)
    fn_sum = torch.zeros(num_classes, device=device)
    tn_sum = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating", leave=False):
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)
            processed_samples += images.size(0)

            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, masks)
            
            # IoU computation - FIX: Sum across batch dimension
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, masks, mode='multiclass', num_classes=num_classes
            )
            
            # Ensure tensors are on the same device and sum across batch dimension
            tp = tp.to(device).sum(dim=0)  # Sum across batch dimension
            fp = fp.to(device).sum(dim=0)  # Sum across batch dimension
            fn = fn.to(device).sum(dim=0)  # Sum across batch dimension
            tn = tn.to(device).sum(dim=0)  # Sum across batch dimension
            
            # Accumulate with proper tensor operations
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
            tn_sum += tn

    eval_loss = running_loss / processed_samples
    final_miou = smp.metrics.iou_score(tp_sum, fp_sum, fn_sum, tn_sum, reduction="micro").item()
    final_accuracy = accuracy_metric.compute().item()

    # Per-class IoU
    per_class_iou = smp.metrics.iou_score(tp_sum, fp_sum, fn_sum, tn_sum, reduction=None)
    
    metrics = {
        "loss": eval_loss,
        "mean_iou": final_miou,
        "pixel_accuracy": final_accuracy,
        "per_class_iou": per_class_iou.tolist() if torch.is_tensor(per_class_iou) else per_class_iou
    }

    accuracy_metric.reset()
    return metrics

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

# --- Training History Visualization ---
def plot_training_history(history, output_dir):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # IoU
    axes[1, 0].plot(history['val_miou'], label='Validation mIoU')
    axes[1, 0].set_title('Validation Mean IoU')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('mIoU')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['learning_rate'], label='Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Main Function ---
def main(args):
    config = load_config(args.config_path)
    seg_config = config['downstream_tasks']['segmentation']
    num_classes = seg_config['num_classes']
    encoder_name = seg_config['encoder']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data Loading
    metadata_df = pd.read_csv(args.metadata_path)
    base_dir = Path(args.base_dir)

    # Split validation from training data
    if args.mode == "baseline":
        train_df = metadata_df[metadata_df['split'] == 'train']
        # Create validation split (20% of training data)
        val_size = int(0.2 * len(train_df))
        val_df = train_df.sample(n=val_size, random_state=42)
        train_df = train_df.drop(val_df.index)
        
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = test_image_dir = val_image_dir = ""
        use_train_generated = use_val_generated = False
        
        print(f"Baseline: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        
    elif args.mode == "generated":
        all_data = metadata_df
        val_size = int(0.2 * len(all_data))
        val_df = all_data.sample(n=val_size, random_state=42)
        train_df = all_data.drop(val_df.index)
        
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = val_image_dir = args.generated_images_dir
        test_image_dir = ""
        use_train_generated = use_val_generated = True
        
        print(f"Generated: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    else:
        raise ValueError("Mode must be 'baseline' or 'generated'")

    # Enhanced augmentations
    img_size = tuple(config.get('image_size', [256, 256]))
    train_augmentation = get_training_augmentation(img_size, encoder_name)
    eval_augmentation = get_validation_testing_augmentation(img_size, encoder_name)

    # Create datasets
    train_dataset = SegmentationDataset(
        train_df, train_image_dir, "", train_augmentation, 
        use_train_generated, base_dir=args.base_dir
    )
    val_dataset = SegmentationDataset(
        val_df, val_image_dir, "", eval_augmentation, 
        use_val_generated, base_dir=args.base_dir
    )
    test_dataset = SegmentationDataset(
        test_df, test_image_dir, "", eval_augmentation, 
        False, base_dir=args.base_dir
    )

    # Data loaders
    batch_size = seg_config.get('batch_size', 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Enhanced model
    model = get_segmentation_model(
        model_name=seg_config['model'],
        encoder=encoder_name,
        num_classes=num_classes
    ).to(device)
    
    # Enhanced loss function
    criterion = CombinedLoss(num_classes=num_classes)
    
    # Enhanced optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=seg_config['learning_rate'],
        weight_decay=1e-4
    )
    
    # More conservative learning rate scheduling
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=5,  # Restart every 5 epochs
        T_mult=2,  # Double the restart interval
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, min_delta=0.001)
    
    # Training history
    history = defaultdict(list)
    
    # Training loop with validation
    print("Starting enhanced training...")
    best_val_miou = 0.0
    
    for epoch in range(seg_config['num_epochs']):
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device, num_classes)
        
        # Scheduler step
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_miou'].append(val_metrics['miou'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{seg_config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['miou']:.4f}, LR: {current_lr:.2e}")
        
        # Save best model
        if val_metrics['miou'] > best_val_miou:
            best_val_miou = val_metrics['miou']
            if args.save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                best_model_path = os.path.join(args.output_dir, f"best_{args.mode}_model.pth")
                torch.save(model.state_dict(), best_model_path)
        
        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Plot training history
    os.makedirs(args.output_dir, exist_ok=True)
    plot_training_history(history, args.output_dir)
    
    # Final evaluation
    print("Starting final evaluation...")
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    
    print("-" * 50)
    print("FINAL TEST RESULTS:")
    print(f"Loss: {test_metrics['loss']:.4f}")
    print(f"Mean IoU: {test_metrics['mean_iou']:.4f}")
    print(f"Pixel Accuracy: {test_metrics['pixel_accuracy']:.4f}")
    print("Per-class IoU:")
    class_names = seg_config.get('class_names', [f"Class {i}" for i in range(num_classes)])
    for i, (name, iou_value) in enumerate(zip(class_names, test_metrics['per_class_iou'])):
        # Check if the IoU value is a list and extract the value if needed
        if isinstance(iou_value, list):
            if len(iou_value) > 0:
                iou_value = iou_value[0]  # Extract first value if it's a list
            else:
                iou_value = 0.0  # Default if empty list
        # Now format as float
        print(f"  {name}: {float(iou_value):.4f}")
    print("-" * 50)

    # Save results
    results = {
        'test_metrics': test_metrics,
        'training_history': dict(history),
        'best_val_miou': best_val_miou,
        'config': seg_config
    }
    
    results_path = os.path.join(args.output_dir, "segmentation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segmentation Training")
    parser.add_argument("--config-path", required=True, help="Path to config YAML")
    parser.add_argument("--metadata-path", required=True, help="Path to metadata CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--mode", required=True, choices=["baseline", "generated"])
    parser.add_argument("--generated-images-dir", help="Generated images directory")
    parser.add_argument("--base-dir", default=".", help="Base directory")
    parser.add_argument("--save-model", action="store_true", help="Save best model")

    args = parser.parse_args()
    
    if args.mode == "generated" and not args.generated_images_dir:
        parser.error("--generated-images-dir required for generated mode")

    main(args)
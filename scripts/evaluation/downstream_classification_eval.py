#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision import models as torchvision_models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import torchmetrics
import numpy as np
from collections import Counter

# --- Configuration ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Dataset ---
class ClassificationDataset(Dataset):
    def __init__(self, metadata_df, image_dir, transform=None, use_generated=False, generated_suffix=".png", base_dir="."):
        self.metadata = metadata_df.reset_index(drop=True)  # Reset index for proper indexing
        self.image_dir = Path(base_dir) / image_dir
        self.transform = transform
        self.use_generated = use_generated
        self.generated_suffix = generated_suffix
        self.base_dir = Path(base_dir)
        
        # Filter out samples with missing images during initialization
        self._filter_valid_samples()
        
    def _filter_valid_samples(self):
        """Filter out samples where images don't exist"""
        valid_indices = []
        for idx in range(len(self.metadata)):
            row = self.metadata.iloc[idx]
            image_id = row['image_id']
            
            if self.use_generated:
                img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
            else:
                img_path_relative = row['real_image_path']
                img_path = self.base_dir / img_path_relative
                
            if img_path.exists():
                valid_indices.append(idx)
            else:
                print(f"Warning: Image not found at {img_path}. Removing from dataset.")
                
        self.metadata = self.metadata.iloc[valid_indices].reset_index(drop=True)
        print(f"Filtered dataset: {len(self.metadata)} valid samples")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label = int(row['classification_label'])

        if self.use_generated:
            img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
        else:
            img_path_relative = row['real_image_path']
            img_path = self.base_dir / img_path_relative

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                image = transforms.ToTensor()(Image.new('RGB', (224, 224), (0, 0, 0)))

        return image, label

# --- Improved Model with Better Architecture ---
def get_model(model_name, num_classes, pretrained=True, dropout_rate=0.5):
    if hasattr(torchvision_models, model_name):
        model = getattr(torchvision_models, model_name)(pretrained=pretrained)
        
        # Modify the final layer for the number of classes with improved architecture
        if hasattr(model, 'fc'):
            num_ftrs = model.fc.in_features
            # Add a more sophisticated classifier head
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(num_ftrs, num_ftrs // 2),
                nn.ReLU(),
                nn.BatchNorm1d(num_ftrs // 2),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(num_ftrs // 2, num_classes)
            )
        elif hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Sequential):
                last_layer_idx = -1
                while not isinstance(model.classifier[last_layer_idx], nn.Linear):
                    last_layer_idx -= 1
                num_ftrs = model.classifier[last_layer_idx].in_features
                model.classifier[last_layer_idx] = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(num_ftrs, num_ftrs // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_ftrs // 2),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(num_ftrs // 2, num_classes)
                )
            elif isinstance(model.classifier, nn.Linear):
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(num_ftrs, num_ftrs // 2),
                    nn.ReLU(),
                    nn.BatchNorm1d(num_ftrs // 2),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(num_ftrs // 2, num_classes)
                )
        else:
            raise AttributeError(f"Model {model_name} structure not recognized for modification.")
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    return model

# --- Training with Gradient Clipping ---
def train_epoch(model, dataloader, criterion, optimizer, device, clip_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Progress bar
    pbar = tqdm(dataloader, desc="Training", leave=False)

    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# --- Evaluation ---
def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    eval_loss = running_loss / len(dataloader.dataset)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    
    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    # AUC calculation
    try:
        if num_classes == 2:
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = 0.0

    metrics = {
        "loss": eval_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc": auc,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
    }

    return metrics

# --- Improved Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_score, model):
        if self.mode == 'min':
            score = -val_score
        else:
            score = val_score
            
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score > self.best_score + self.min_delta:
            self.best_score = score
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

# --- Improved Class Weight Calculation ---
def compute_class_weights(labels, method='balanced'):
    """Compute class weights for imbalanced datasets"""
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    if method == 'balanced':
        # sklearn's balanced method
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = total_samples / (num_classes * count)
    elif method == 'inverse':
        # Simple inverse frequency
        weights = {}
        for class_id, count in class_counts.items():
            weights[class_id] = 1.0 / count
    else:
        # Uniform weights
        weights = {class_id: 1.0 for class_id in class_counts.keys()}
    
    # Convert to tensor
    weight_tensor = torch.zeros(num_classes)
    for class_id, weight in weights.items():
        weight_tensor[class_id] = weight
    
    # Normalize
    weight_tensor = weight_tensor / weight_tensor.sum() * num_classes
    
    return weight_tensor

# --- Main Function ---
def main(args):
    config = load_config(args.config_path)
    cls_config = config['downstream_tasks']['classification']
    num_classes = cls_config['num_classes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    metadata_df = pd.read_csv(args.metadata_path)
    base_dir = Path(args.base_dir)

    # Create proper train/validation/test splits
    if args.mode == "baseline":
        original_train_df = metadata_df[metadata_df['split'] == 'train']
        test_df = metadata_df[metadata_df['split'] == 'test']
        
        # Stratified split to maintain class distribution
        train_df, val_df = train_test_split(
            original_train_df, 
            test_size=0.2,
            stratify=original_train_df['classification_label'],
            random_state=42
        )
        
        train_image_dir = ""
        val_image_dir = ""
        test_image_dir = ""
        use_train_generated = False
        use_val_generated = False
        
        print(f"Running Baseline:")
        print(f"  Training on {len(train_df)} real samples")
        print(f"  Validation on {len(val_df)} real samples") 
        print(f"  Testing on {len(test_df)} real samples")
        
    elif args.mode == "generated":
        original_train_df = metadata_df[metadata_df['split'] == 'train']
        test_df = metadata_df[metadata_df['split'] == 'test']
        
        # Create validation set from original train data (real images)
        train_df_real, val_df = train_test_split(
            original_train_df,
            test_size=0.2,
            stratify=original_train_df['classification_label'],
            random_state=42
        )
        
        # Use ALL data for training on generated images
        train_df = metadata_df
        
        train_image_dir = args.generated_images_dir
        val_image_dir = ""
        test_image_dir = ""
        use_train_generated = True
        use_val_generated = False
        
        print(f"Running Generated Eval:")
        print(f"  Training on {len(train_df)} generated samples")
        print(f"  Validation on {len(val_df)} real samples")
        print(f"  Testing on {len(test_df)} real samples")
    else:
        raise ValueError("Mode must be 'baseline' or 'generated'")

    # Improved transforms
    img_size = tuple(config.get('image_size', [224, 224]))
    
    # More aggressive training augmentation
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size[0] * 1.1), int(img_size[1] * 1.1))),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)
    ])
    
    # Validation/test transforms
    eval_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = ClassificationDataset(train_df, train_image_dir, transform=train_transform, use_generated=use_train_generated, base_dir=args.base_dir)
    val_dataset = ClassificationDataset(val_df, val_image_dir, transform=eval_transform, use_generated=use_val_generated, base_dir=args.base_dir)
    test_dataset = ClassificationDataset(test_df, test_image_dir, transform=eval_transform, use_generated=False, base_dir=args.base_dir)

    # Improved class weight computation
    train_labels = [int(row['classification_label']) for _, row in train_df.iterrows()]
    class_weights = compute_class_weights(train_labels, method='balanced')
    print(f"Class distribution: {Counter(train_labels)}")
    print(f"Class weights: {class_weights}")

    # Create dataloaders without weighted sampling (let class weights in loss handle imbalance)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cls_config['batch_size'], 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    val_loader = DataLoader(val_dataset, batch_size=cls_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cls_config['batch_size'], shuffle=False, num_workers=4, pin_memory=True)

    # --- Model Initialization ---
    model = get_model(cls_config['model'], num_classes, dropout_rate=0.5).to(device)
    
    # Initialize weights properly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    # Apply weight initialization to new layers only
    for module in model.modules():
        if hasattr(module, 'fc') and isinstance(module.fc, nn.Sequential):
            module.fc.apply(init_weights)
        elif hasattr(module, 'classifier') and isinstance(module.classifier, nn.Sequential):
            module.classifier.apply(init_weights)

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Improved optimizer with different learning rates for different parts
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': cls_config['learning_rate'] * 0.1},  # Lower LR for pretrained features
        {'params': classifier_params, 'lr': cls_config['learning_rate']}       # Higher LR for new classifier
    ], weight_decay=1e-4)

    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cls_config['num_epochs'], eta_min=1e-6
    )
    
    # Early stopping based on validation F1 score
    early_stopping = EarlyStopping(patience=10, min_delta=0.001, mode='max')

    # --- Training Loop ---
    print("Starting training...")
    best_val_f1 = 0.0
    training_history = []
    
    for epoch in range(cls_config['num_epochs']):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        
        # Update learning rate
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch results
        epoch_info = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['accuracy'],
            'val_f1': val_metrics['f1_score'],
            'learning_rate': current_lr
        }
        training_history.append(epoch_info)
        
        print(f"Epoch {epoch+1}/{cls_config['num_epochs']} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, "
              f"Val F1: {val_metrics['f1_score']:.4f}, LR: {current_lr:.2e}")
        
        # Track best validation F1
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            
        # Early stopping based on validation F1
        if early_stopping(val_metrics['f1_score'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # --- Final Evaluation on Test Set ---
    print("Starting final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print("-" * 50)
    print("Final Test Set Evaluation Results:")
    for key, value in test_metrics.items():
        if isinstance(value, list):
            print(f"{key.capitalize()}: {value}")
        else:
            print(f"{key.capitalize()}: {value:.4f}")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print("-" * 50)

    # --- Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "classification_results.json")
    
    results = {
        "test_metrics": test_metrics,
        "best_val_f1": best_val_f1,
        "mode": args.mode,
        "training_history": training_history,
        "final_class_distribution": dict(Counter(train_labels)),
        "class_weights_used": class_weights.tolist()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Classification results saved to {results_path}")

    # Save the trained model
    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{args.mode}_classification_model.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cls_config,
            'class_weights': class_weights,
            'best_val_f1': best_val_f1,
            'test_metrics': test_metrics
        }, model_path)
        print(f"Trained model saved to {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream Classification Evaluation")
    parser.add_argument("--config-path", required=True, help="Path to the main evaluation config YAML.")
    parser.add_argument("--metadata-path", required=True, help="Path to the downstream metadata CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory to save evaluation results.")
    parser.add_argument("--mode", required=True, choices=["baseline", "generated"], help="Evaluation mode: 'baseline' (train/test on real) or 'generated' (train on generated/test on real).")
    parser.add_argument("--generated-images-dir", help="Directory containing generated images (required if mode='generated'). Relative to base_dir.")
    parser.add_argument("--base-dir", default=".", help="Project base directory for resolving relative paths.")
    parser.add_argument("--save-model", action="store_true", help="Save the trained model.")

    args = parser.parse_args()

    if args.mode == "generated" and not args.generated_images_dir:
        parser.error("--generated-images-dir is required when mode='generated'")

    main(args)
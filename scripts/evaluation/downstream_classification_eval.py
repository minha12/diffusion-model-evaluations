#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import yaml
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this import
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision import models as torchvision_models
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import json
import torchmetrics # Use torchmetrics for consistency

# --- Configuration ---
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Dataset ---
class ClassificationDataset(Dataset):
    def __init__(self, metadata_df, image_dir, transform=None, use_generated=False, generated_suffix=".png", base_dir="."):
        self.metadata = metadata_df
        self.image_dir = Path(base_dir) / image_dir # Make image_dir relative to base_dir
        self.transform = transform
        self.use_generated = use_generated
        self.generated_suffix = generated_suffix # Suffix of generated images
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        image_id = row['image_id']
        label = int(row['classification_label'])

        if self.use_generated:
            # Assumes generated image filenames match image_id + suffix
            img_path = self.image_dir / f"{image_id}{self.generated_suffix}"
        else:
            # Use the real image path from metadata, making it absolute if needed
            img_path_relative = row['real_image_path']
            img_path = self.base_dir / img_path_relative # Construct absolute path


        if not img_path.exists():
             print(f"Warning: Image not found at {img_path}. Skipping sample.")
             # Return dummy data or raise error, depending on desired handling
             # For now, returning None to be filtered later in dataloader collate_fn if needed
             return None, None

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None, None # Handle error

        return image, label

# --- Model ---
def get_model(model_name, num_classes, pretrained=True):
    if hasattr(torchvision_models, model_name):
        model = getattr(torchvision_models, model_name)(pretrained=pretrained)
        # Modify the final layer for the number of classes
        if hasattr(model, 'fc'): # For ResNet, etc.
            num_ftrs = model.fc.in_features
            # Replace fc with a sequence that includes dropout
            model.fc = nn.Sequential(
                nn.Dropout(0.1),  # Reduce from 0.5 to 0.3
                nn.Linear(num_ftrs, num_classes)
            )
        elif hasattr(model, 'classifier'): # For VGG, DenseNet, etc.
             if isinstance(model.classifier, nn.Sequential):
                 # Find the last linear layer
                 last_layer_idx = -1
                 while not isinstance(model.classifier[last_layer_idx], nn.Linear):
                     last_layer_idx -= 1
                 num_ftrs = model.classifier[last_layer_idx].in_features
                 
                 # Insert dropout before the last layer
                 model.classifier = nn.Sequential(
                     *list(model.classifier[:last_layer_idx]),
                     nn.Dropout(0.1),
                     nn.Linear(num_ftrs, num_classes)
                 )
             elif isinstance(model.classifier, nn.Linear): # Single classifier layer
                 num_ftrs = model.classifier.in_features
                 model.classifier = nn.Sequential(
                     nn.Dropout(0.1),
                     nn.Linear(num_ftrs, num_classes)
                 )
        else:
            raise AttributeError(f"Model {model_name} structure not recognized for modification.")
    else:
        raise ValueError(f"Model {model_name} not found in torchvision.models")
    return model

# --- Training ---
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        # Filter out None items caused by loading errors
        valid_indices = [i for i, (img, lbl) in enumerate(zip(inputs, labels)) if img is not None]
        if not valid_indices:
            continue
        inputs = torch.stack([inputs[i] for i in valid_indices]).to(device)
        labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.long).to(device) # Ensure labels are Long type

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

# --- Evaluation ---
def evaluate(model, dataloader, criterion, device, num_classes):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = [] # For AUC

    # Initialize TorchMetrics
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    # Use average='macro' for balanced precision/recall/F1 if classes are imbalanced
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average='macro').to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average='macro').to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average='macro').to(device)
    auc_metric = torchmetrics.AUROC(task="multiclass", num_classes=num_classes).to(device) if num_classes > 1 else None # AUROC needs >1 class

    # Add per-class metrics
    per_class_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average=None).to(device)
    per_class_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average=None).to(device)
    per_class_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average=None).to(device)


    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            # Filter out None items
            valid_indices = [i for i, (img, lbl) in enumerate(zip(inputs, labels)) if img is not None]
            if not valid_indices:
                 continue
            inputs = torch.stack([inputs[i] for i in valid_indices]).to(device)
            labels = torch.tensor([labels[i] for i in valid_indices], dtype=torch.long).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            # Update metrics
            accuracy_metric.update(preds, labels)
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            if auc_metric:
                 auc_metric.update(probs, labels) # AUROC needs probabilities

            # Update per-class metrics
            per_class_precision.update(preds, labels)
            per_class_recall.update(preds, labels)
            per_class_f1.update(preds, labels)

            # Store for sklearn metrics if needed (optional, as torchmetrics is preferred)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())


    eval_loss = running_loss / len(dataloader.dataset)

    # Compute final metrics
    final_accuracy = accuracy_metric.compute().item()
    final_precision = precision_metric.compute().item()
    final_recall = recall_metric.compute().item()
    final_f1 = f1_metric.compute().item()
    final_auc = auc_metric.compute().item() if auc_metric else 0.0 # Handle binary case or if AUC fails

    # Compute per-class metrics
    per_class_precision_values = per_class_precision.compute().cpu().numpy()
    per_class_recall_values = per_class_recall.compute().cpu().numpy()
    per_class_f1_values = per_class_f1.compute().cpu().numpy()

    metrics = {
        "loss": eval_loss,
        "accuracy": final_accuracy,
        "precision": final_precision,
        "recall": final_recall,
        "f1_score": final_f1,
        "auc": final_auc,
        "per_class_precision": per_class_precision_values.tolist(),
        "per_class_recall": per_class_recall_values.tolist(),
        "per_class_f1": per_class_f1_values.tolist(),
    }

    # Reset metrics for next evaluation
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    if auc_metric: auc_metric.reset()
    per_class_precision.reset()
    per_class_recall.reset()
    per_class_f1.reset()

    return metrics


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weight for each class
        self.gamma = gamma  # Focusing parameter
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# --- Main Function ---
def main(args):
    class_counts = [float(x) for x in args.class_counts.split(',')]
    config = load_config(args.config_path)
    cls_config = config['downstream_tasks']['classification']
    num_classes = cls_config['num_classes']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading ---
    metadata_df = pd.read_csv(args.metadata_path)
    # Define base directory for resolving relative paths
    base_dir = Path(args.base_dir)

    # Print class distribution information
    print(f"Using class distribution: {class_counts}")
    
    # Determine train/test data based on mode
    if args.mode == "baseline":
        # Train on real train split, test on real test split
        train_df = metadata_df[metadata_df['split'] == 'train']
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = "" # Paths are absolute/relative in df
        test_image_dir = "" # Paths are absolute/relative in df
        use_train_generated = False
        print(f"Running Baseline: Training on {len(train_df)} real samples, Testing on {len(test_df)} real samples.")
    elif args.mode == "generated":
        # Train on generated images (corresponding to all real data), test on real test split
        train_df = metadata_df # Use all metadata for mapping
        test_df = metadata_df[metadata_df['split'] == 'test']
        train_image_dir = args.generated_images_dir # Relative path from base_dir
        test_image_dir = "" # Paths are absolute/relative in df
        use_train_generated = True
        print(f"Running Generated Eval: Training on {len(train_df)} generated samples, Testing on {len(test_df)} real samples.")
    else:
        raise ValueError("Mode must be 'baseline' or 'generated'")

    # Define transforms (adjust as needed)
    img_size = tuple(config.get('image_size', [256, 256])) # Get image size from main config if available
    transform_train = transforms.Compose([
        transforms.Resize((img_size[0] + 20, img_size[1] + 20)),  # Resize larger
        transforms.RandomCrop(img_size),  # Then crop to desired size
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Keep simpler transform for test data
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Create datasets
    train_dataset = ClassificationDataset(train_df, train_image_dir, transform=transform_train, use_generated=use_train_generated, base_dir=args.base_dir)
    test_dataset = ClassificationDataset(test_df, test_image_dir, transform=transform_test, use_generated=False, base_dir=args.base_dir) # Always test on real

    # Compute sample weights (for each data point based on its class)
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    class_weights = class_weights / class_weights.sum()  # Normalize weights
    print(f"Class weights: {class_weights}")
    
    sample_weights = [class_weights[label] for label in train_df['classification_label']]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Create dataloaders
    # Handle potential None items from dataset loading errors
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None and x[0] is not None, batch))
        if not batch: return torch.Tensor(), torch.Tensor() # Return empty tensors if batch is empty
        return torch.utils.data.dataloader.default_collate(batch)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cls_config['batch_size'], 
        sampler=sampler,  # Use sampler instead of shuffle
        num_workers=4, 
        collate_fn=collate_fn
    )
    test_loader = DataLoader(test_dataset, batch_size=cls_config['batch_size'], shuffle=False, num_workers=4, collate_fn=collate_fn)

    # --- Model Initialization ---
    model = get_model(cls_config['model'], num_classes).to(device)

    # Calculate weights inversely proportional to class frequencies
    weights = class_weights.to(device)  # Reuse the same weights calculated earlier

    # Use weighted CrossEntropyLoss
    criterion = FocalLoss(alpha=weights, gamma=2.0)

    optimizer = optim.Adam(model.parameters(), 
                      lr=cls_config['learning_rate'], 
                      weight_decay=0.0001)  # Reduce from 0.001 to 0.0005

    # --- Learning Rate Scheduler ---
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    # --- Training Loop ---
    best_f1 = 0
    patience = 5
    patience_counter = 0
    best_model_state = None

    print("Starting training...")
    for epoch in range(cls_config['num_epochs']):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{cls_config['num_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        # Validation
        val_metrics = evaluate(model, test_loader, criterion, device, num_classes)
        current_f1 = val_metrics['f1_score']
        print(f"Validation F1: {current_f1:.4f}")
        
        # Learning Rate Scheduling
        scheduler.step(current_f1)
        
        # Early stopping logic
        if current_f1 > best_f1:
            best_f1 = current_f1
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"New best F1: {best_f1:.4f}")
        else:
            patience_counter += 1
            print(f"F1 did not improve. Patience: {patience_counter}/{patience}")
            
        if patience_counter >= patience:
            print("Early stopping triggered!")
            break

    # Load best model before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Loaded best model for final evaluation")

    # --- Final Evaluation ---
    print("Starting final evaluation on test set...")
    test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
    print("-" * 30)
    print("Test Set Evaluation Results:")
    for key, value in test_metrics.items():
        if isinstance(value, list):
            print(f"{key.capitalize()}: {value}")
        else:
            print(f"{key.capitalize()}: {value:.4f}")
    print("-" * 30)

    # --- Save Results ---
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "classification_results.json")
    with open(results_path, 'w') as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Classification results saved to {results_path}")

    # Optionally save the trained model
    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{args.mode}_classification_model.pth")
        torch.save(model.state_dict(), model_path)
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
    parser.add_argument("--class-counts", default="19.49,7.49,73.02", 
                      help="Comma-separated class distribution percentages (default: 19.49,7.49,73.02)")

    args = parser.parse_args()

    if args.mode == "generated" and not args.generated_images_dir:
        parser.error("--generated-images-dir is required when mode='generated'")

    main(args)
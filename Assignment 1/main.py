import os
import random
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# ==========================================
# 1. Configurations
# ==========================================
BATCH_SIZE = 64
EPOCHS = 15
LR = 0.001
SEED = 42
OUTPUT_DIR = 'output'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 normalization (per-channel mean and std)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

# Sub-directories
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PLOTS_DIR = os.path.join(OUTPUT_DIR, 'plots')
GRADCAM_DIR = os.path.join(OUTPUT_DIR, 'gradcam')
GRADCAM_BASELINE_DIR = os.path.join(GRADCAM_DIR, 'baseline')
GRADCAM_SE_DIR = os.path.join(GRADCAM_DIR, 'se_attention')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')

# ==========================================
# 2. Utilities
# ==========================================
def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dirs():
    """Creates all output directories."""
    for d in [MODELS_DIR, PLOTS_DIR, GRADCAM_BASELINE_DIR, GRADCAM_SE_DIR, RESULTS_DIR]:
        os.makedirs(d, exist_ok=True)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plots and saves training curves."""
    epochs_range = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, label='Train Acc')
    plt.plot(epochs_range, val_accs, label='Val Acc')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_class_wise_accuracy(model, loader, device, classes):
    """Calculates accuracy for each class individually."""
    model.eval()
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predictions = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label.item()]] += 1
                total_pred[classes[label.item()]] += 1

    accuracies = {}
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] > 0:
            accuracy = 100 * float(correct_count) / total_pred[classname]
        else:
            accuracy = 0.0
        accuracies[classname] = accuracy
        
    return accuracies

def save_accuracy_json(accuracies, save_path):
    """Saves class-wise accuracy dict to a JSON file."""
    with open(save_path, 'w') as f:
        json.dump(accuracies, f, indent=4)
    print(f"  Saved accuracy JSON -> {save_path}")

# ==========================================
# 3. Dataset (CIFAR-10)
# ==========================================
def get_dataloaders(batch_size=64, data_dir='./data', val_split=0.1, seed=42, augment=False):
    """
    Downloads and prepares CIFAR-10 dataloaders (Train, Val, Test).
    CIFAR-10: 60k 32x32 color images in 10 classes.
    """
    # Base transform (for Test/Val)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    ])

    # Train transform with augmentation
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
        ])
    else:
        train_transform = base_transform

    # Download datasets
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None 
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=base_transform
    )

    # Split train and validation
    generator = torch.Generator().manual_seed(seed)
    train_size = int((1 - val_split) * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    
    raw_train_subset, raw_val_subset = random_split(
        full_train_dataset, [train_size, val_size], generator=generator
    )
    
    class TransformedSubset(torch.utils.data.Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
            
        def __getitem__(self, index):
            x, y = self.subset[index]
            if self.transform:
                x = self.transform(x)
            return x, y
        
        def __len__(self):
            return len(self.subset)
            
    train_dataset = TransformedSubset(raw_train_subset, transform=train_transform)
    val_dataset = TransformedSubset(raw_val_subset, transform=base_transform)

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # CIFAR-10 class names
    class_names = ('airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, val_loader, test_loader, class_names

# ==========================================
# 4. Models
# ==========================================
class BaselineCNN(nn.Module):
    """
    A custom CNN for CIFAR-10.
    Input: 3x32x32 (RGB)
    """
    def __init__(self, num_classes=10):
        super(BaselineCNN, self).__init__()
        # Block 1: 3 -> 32 channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 32x32 -> 16x16
        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 16x16 -> 8x8
        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        
        # Fully Connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block.
    
    Learns per-channel attention weights:
      1. Squeeze: Global Average Pooling reduces HxW to 1x1
      2. Excitation: Two FC layers learn channel importance (bottleneck -> expand)
      3. Scale: Reweight original feature maps by learned channel attention
    
    Reference: Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """
    def __init__(self, channels, reduction=4):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)  # Global Avg Pool -> (N, C, 1, 1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        N, C, _, _ = x.size()
        # Squeeze: (N, C, H, W) -> (N, C)
        y = self.squeeze(x).view(N, C)
        # Excitation: (N, C) -> (N, C) channel attention weights
        y = self.excitation(y).view(N, C, 1, 1)
        # Scale: element-wise multiply
        return x * y.expand_as(x)


class SE_CNN(nn.Module):
    """
    Baseline CNN enhanced with Squeeze-and-Excitation (SE) attention blocks.
    Same conv architecture as BaselineCNN, but each conv block is followed by
    an SE block that learns to reweight feature map channels by importance.
    Input: 3x32x32 (RGB)
    """
    def __init__(self, num_classes=10):
        super(SE_CNN, self).__init__()
        # Block 1: Conv + BN + SE
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.se1 = SEBlock(32, reduction=4)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 2: Conv + BN + SE
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.se2 = SEBlock(64, reduction=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Block 3: Conv + BN + SE
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.se3 = SEBlock(128, reduction=4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(self.se1(x))       # SE attention after conv1
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(self.se2(x))       # SE attention after conv2
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(self.se3(x))       # SE attention after conv3
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 5. Explainability (Grad-CAM)
# ==========================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handle_fwd = self.target_layer.register_forward_hook(self.save_activation)
        self.handle_bwd = self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        target = output[0][target_class]
        target.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        activations = np.maximum(activations, 0)
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (32, 32))  # CIFAR-10 is 32x32
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        return cam, target_class

    def remove_hooks(self):
        self.handle_fwd.remove()
        self.handle_bwd.remove()

# ==========================================
# 6. Training Loop
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / total, 100 * correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return running_loss / total, 100 * correct / total

def train_loop(model, train_loader, val_loader, criterion, optimizer, epochs, device, save_path):
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print(f"  Starting training on {device}...")
    
    for epoch in range(epochs):
        start_time = time.time()
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        
        elapsed = time.time() - start_time
        print(f"  Epoch {epoch+1}/{epochs} - {elapsed:.1f}s - "
              f"Train Loss: {t_loss:.4f}, Acc: {t_acc:.2f}% | "
              f"Val Loss: {v_loss:.4f}, Acc: {v_acc:.2f}%")
        
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(model.state_dict(), save_path)
            print(f"    -> Best model saved (Val Acc: {best_acc:.2f}%)")

    print(f"  Training complete. Best Val Acc: {best_acc:.2f}%")
    return train_losses, val_losses, train_accs, val_accs

# ==========================================
# 7. Main Pipeline
# ==========================================
def denormalize_cifar(img_tensor):
    """Denormalize a CIFAR-10 image tensor back to [0,1] for display."""
    mean = torch.tensor(CIFAR10_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()  # (H, W, 3)

def main():
    print(f"=== Baseline vs SE-Attention CNN on CIFAR-10 (Device: {DEVICE}) ===")
    set_seed(SEED)
    create_dirs()
    
    # --- Data ---
    print("\nLoading CIFAR-10...")
    train_loader, val_loader, test_loader, classes = get_dataloaders(BATCH_SIZE, augment=True)
    print(f"  Classes: {classes}")
    
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # PART 1: Baseline CNN
    # -------------------------
    print("\n--- [1/2] Baseline CNN ---")
    baseline_path = os.path.join(MODELS_DIR, 'baseline_model.pth')
    baseline_model = BaselineCNN(num_classes=10).to(DEVICE)
    
    if os.path.exists(baseline_path):
        print("  Found saved baseline model, loading...")
        baseline_model.load_state_dict(torch.load(baseline_path, map_location=DEVICE, weights_only=True))
    else:
        optimizer = optim.Adam(baseline_model.parameters(), lr=LR)
        t_l, v_l, t_a, v_a = train_loop(
            baseline_model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, baseline_path
        )
        plot_training_curves(t_l, v_l, t_a, v_a, os.path.join(PLOTS_DIR, 'baseline_curves.png'))

    # -------------------------
    # PART 2: SE-Attention CNN
    # -------------------------
    print("\n--- [2/2] SE-Attention CNN (Squeeze-and-Excitation) ---")
    se_path = os.path.join(MODELS_DIR, 'se_attention_model.pth')
    se_model = SE_CNN(num_classes=10).to(DEVICE)
    
    if os.path.exists(se_path):
        print("  Found saved SE model, loading...")
        se_model.load_state_dict(torch.load(se_path, map_location=DEVICE, weights_only=True))
    else:
        optimizer = optim.Adam(se_model.parameters(), lr=LR)
        t_l, v_l, t_a, v_a = train_loop(
            se_model, train_loader, val_loader, criterion, optimizer, EPOCHS, DEVICE, se_path
        )
        plot_training_curves(t_l, v_l, t_a, v_a, os.path.join(PLOTS_DIR, 'se_attention_curves.png'))

    # -------------------------
    # PART 3: Evaluation
    # -------------------------
    print("\n--- Evaluation & Analysis ---")
    baseline_model.eval()
    se_model.eval()
    
    # Class-wise accuracy
    base_accs = get_class_wise_accuracy(baseline_model, test_loader, DEVICE, classes)
    se_accs = get_class_wise_accuracy(se_model, test_loader, DEVICE, classes)
    
    save_accuracy_json(base_accs, os.path.join(RESULTS_DIR, 'baseline_classwise_accuracy.json'))
    save_accuracy_json(se_accs, os.path.join(RESULTS_DIR, 'se_attention_classwise_accuracy.json'))
    
    # Print comparison table
    print(f"\n{'Class':<15} | {'Baseline':<10} | {'SE-Attn':<10} | {'Change':<10}")
    print("-" * 55)
    overall_b, overall_s = 0, 0
    
    for cls in classes:
        b = base_accs[cls]
        s = se_accs[cls]
        print(f"{cls:<15} | {b:.2f}%     | {s:.2f}%     | {s-b:+.2f}%")
        overall_b += b
        overall_s += s
    
    print("-" * 55)
    print(f"{'Overall':<15} | {overall_b/10:.2f}%     | {overall_s/10:.2f}%     | {(overall_s-overall_b)/10:+.2f}%")

    # Bar Chart
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, [base_accs[c] for c in classes], width, label='Baseline', color='#ff9999')
    ax.bar(x + width/2, [se_accs[c] for c in classes], width, label='SE-Attention', color='#66b3ff')
    ax.set_title('Per-Class Accuracy: Baseline vs SE-Attention CNN (CIFAR-10)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'accuracy_comparison.png'))
    plt.close()
    print(f"  Saved accuracy comparison -> {os.path.join(PLOTS_DIR, 'accuracy_comparison.png')}")

    # Confusion Matrices
    def plot_cm(model, name):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(DEVICE)
                _, preds = torch.max(model(inputs), 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
        plt.title(f'Confusion Matrix: {name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, f'confusion_matrix_{name.lower().replace(" ", "_").replace("-", "_")}.png'))
        plt.close()

    plot_cm(baseline_model, 'Baseline')
    plot_cm(se_model, 'SE-Attention')
    print(f"  Saved confusion matrices -> {PLOTS_DIR}")

    # -------------------------
    # PART 4: Grad-CAM
    # -------------------------
    print("\n--- Grad-CAM Visualizations ---")
    grad_cam_base = GradCAM(baseline_model, baseline_model.conv3)
    grad_cam_se = GradCAM(se_model, se_model.conv3)
    
    # CIFAR-10 interesting classes for Grad-CAM comparison
    targets = ['cat', 'dog', 'bird', 'airplane', 'ship']
    found = {k: 0 for k in targets}
    samples_per_class = 2
    
    for i, (img, label) in enumerate(test_loader.dataset):
        cls_name = classes[label]
        if cls_name in targets and found[cls_name] < samples_per_class:
            input_tensor = img.unsqueeze(0).to(DEVICE)
            cam_b, _ = grad_cam_base.generate_cam(input_tensor, label)
            cam_s, _ = grad_cam_se.generate_cam(input_tensor, label)
            
            # Denormalize for display (CIFAR-10 is RGB)
            img_disp = denormalize_cifar(img)
            img_rgb = (img_disp * 255).astype(np.uint8)
            
            hb = cv2.applyColorMap((cam_b * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hb = cv2.cvtColor(hb, cv2.COLOR_BGR2RGB)
            hs = cv2.applyColorMap((cam_s * 255).astype(np.uint8), cv2.COLORMAP_JET)
            hs = cv2.cvtColor(hs, cv2.COLOR_BGR2RGB)
            
            overlay_b = cv2.addWeighted(img_rgb, 0.6, hb, 0.4, 0)
            overlay_s = cv2.addWeighted(img_rgb, 0.6, hs, 0.4, 0)
            
            # Save individual Grad-CAM images
            fname = f'{cls_name}_{found[cls_name]}.png'
            plt.imsave(os.path.join(GRADCAM_BASELINE_DIR, fname), overlay_b)
            plt.imsave(os.path.join(GRADCAM_SE_DIR, fname), overlay_s)
            
            # Save combined side-by-side (Original | Baseline | SE-Attention)
            fig, axes = plt.subplots(1, 3, figsize=(9, 3))
            axes[0].imshow(img_disp)
            axes[0].set_title(f'{cls_name}\n(Original)')
            axes[0].axis('off')
            axes[1].imshow(overlay_b)
            axes[1].set_title('Baseline\nGrad-CAM')
            axes[1].axis('off')
            axes[2].imshow(overlay_s)
            axes[2].set_title('SE-Attention\nGrad-CAM')
            axes[2].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'gradcam_{cls_name}_{found[cls_name]}.png'))
            plt.close()
            
            found[cls_name] += 1
            
        if all(v >= samples_per_class for v in found.values()):
            break
    
    grad_cam_base.remove_hooks()
    grad_cam_se.remove_hooks()
    print(f"  Saved Grad-CAM images -> {GRADCAM_DIR}")
    print(f"  Saved Grad-CAM comparison plots -> {PLOTS_DIR}")

    # -------------------------
    # PART 5: Failure Cases
    # -------------------------
    print("\n--- Failure Cases (SE-Attention) ---")
    failures = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = se_model(inputs)
            probs = torch.softmax(outputs, dim=1)
            confs, preds = torch.max(probs, 1)
            
            errors = (preds != labels)
            if errors.any():
                err_indices = torch.where(errors)[0]
                for idx in err_indices:
                    if len(failures) < 5:
                        failures.append({
                            'img': inputs[idx],
                            'true': labels[idx].item(),
                            'pred': preds[idx].item(),
                            'conf': confs[idx].item()
                        })
            if len(failures) >= 5:
                break
                
    if failures:
        fig, axes = plt.subplots(1, len(failures), figsize=(15, 3))
        if len(failures) == 1: axes = [axes]
        for i, fail in enumerate(failures):
            img = denormalize_cifar(fail['img'])
            axes[i].imshow(img)
            axes[i].set_title(f"True: {classes[fail['true']]}\nPred: {classes[fail['pred']]}\nConf: {fail['conf']:.2f}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'failure_cases.png'))
        plt.close()
        print(f"  Saved failure cases -> {os.path.join(PLOTS_DIR, 'failure_cases.png')}")

    # -------------------------
    # Summary
    # -------------------------
    print(f"\n{'='*60}")
    print(f"Pipeline Complete! Output structure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"    models/          - baseline_model.pth, se_attention_model.pth")
    print(f"    plots/           - Training curves, accuracy comparison,")
    print(f"                       confusion matrices, Grad-CAM comparisons")
    print(f"    gradcam/")
    print(f"      baseline/      - Baseline Grad-CAM heatmaps")
    print(f"      se_attention/  - SE-Attention Grad-CAM heatmaps")
    print(f"    results/         - Class-wise accuracy JSONs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
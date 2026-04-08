import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from PIL import Image  # нужен только для конвертации из OpenCV BGR в RGB

def cv2_loader(path):
    """Загружает изображение через OpenCV, конвертирует BGR->RGB и возвращает PIL Image."""
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise OSError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)

# ----------------------------------------------------------------------
# paths/settings
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DIR = os.path.join(BASE_DIR, "data", "binary", "train")
VAL_DIR = os.path.join(BASE_DIR, "data", "binary", "val")

os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "dental_binary_model.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 16
NUM_EPOCHS = 15
LEARNING_RATE = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------------------------------------
# transform
# ----------------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ----------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------
train_dataset = datasets.ImageFolder(root=TRAIN_DIR, transform=transform_train, loader=cv2_loader)
val_dataset = datasets.ImageFolder(root=VAL_DIR, transform=transform_val, loader=cv2_loader)

classes = train_dataset.classes
num_classes = len(classes)
print(f"Classes: {classes}")

class_counts = [len(os.listdir(os.path.join(TRAIN_DIR, cls))) for cls in classes]
print(f"Class counts in train: {dict(zip(classes, class_counts))}")

class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
sample_weights = [class_weights[label] for _, label in train_dataset.samples]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------------------------------------------------
# model, loss, optimisation
# ----------------------------------------------------------------------
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

loss_weights = torch.tensor([1.0, class_counts[0]/class_counts[1]], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

# ----------------------------------------------------------------------
# learning cycle
# ----------------------------------------------------------------------
train_losses = []
val_accuracies = []
best_val_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_acc = 100 * correct / total
    val_accuracies.append(val_acc)
    scheduler.step(val_acc)
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.2f}%")

print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}%")

# ----------------------------------------------------------------------
# final grade and graphs
# ----------------------------------------------------------------------
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
plt.figure(figsize=(6,5))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix on Validation Set")
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix_binary.png"))
plt.show()

print("\n--- Classification Report ---")
print(classification_report(all_labels, all_preds, target_names=classes))

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(range(1, NUM_EPOCHS+1), train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1,2,2)
plt.plot(range(1, NUM_EPOCHS+1), val_accuracies, marker='o', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy (%)')
plt.title('Validation Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "training_curves_binary.png"))
plt.show()

print(f"\nModel saved to: {MODEL_SAVE_PATH}")
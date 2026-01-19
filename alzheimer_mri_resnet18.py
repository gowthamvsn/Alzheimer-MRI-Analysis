# ============================================================
# ALZHEIMER'S MRI CLASSIFICATION USING RESNET18
# ============================================================
# End-to-end deep learning pipeline for classifying Alzheimer's
# disease stages from MRI images using transfer learning.
# ============================================================


# ==============================
# IMPORTS & CONFIGURATION
# ==============================
import os
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, fbeta_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Hugging Face dataset
from datasets import load_dataset


# ==============================
# GLOBAL SETTINGS
# ==============================
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("ALZHEIMER'S MRI CLASSIFICATION - RESNET18 PIPELINE")
print(f"Using device: {DEVICE}")
print("=" * 70)


# ==============================
# DATA LOADING
# ==============================
print("\n[STEP 1/6] Loading dataset from Hugging Face...")

dataset = load_dataset("Falah/Alzheimer_MRI")

print("Dataset loaded successfully!")
print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

label_map = {
    "Mild_Demented": 0,
    "Moderate_Demented": 1,
    "Non_Demented": 2,
    "Very_Mild_Demented": 3
}


# ==============================
# DATASET CLASS
# ==============================
class AlzheimerDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = sample["image"]
        label = label_map[sample["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


# ==============================
# TRANSFORMS & DATALOADERS
# ==============================
print("\n[STEP 2/6] Preparing data loaders...")

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

train_dataset = AlzheimerDataset(dataset["train"], train_transforms)
test_dataset = AlzheimerDataset(dataset["test"], test_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("Data loaders ready.")


# ==============================
# MODEL DEFINITION (RESNET18)
# ==============================
print("\n[STEP 3/6] Building ResNet18 model...")

model = models.resnet18(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(label_map))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

print("Model initialized.")


# ==============================
# TRAINING LOOP
# ==============================
print("\n[STEP 4/6] Training model...")

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 30)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)

    print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f}")

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

print("\nTraining complete.")
print(f"Best training accuracy: {best_acc:.4f}")

model.load_state_dict(best_model_wts)


# ==============================
# MODEL EVALUATION
# ==============================
print("\n[STEP 5/6] Evaluating model...")

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=label_map.keys()))

print(f"Test Accuracy: {acc:.4f}")


# ==============================
# CONFUSION MATRIX
# ==============================
print("\n[STEP 6/6] Plotting confusion matrix...")

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_map.keys(),
            yticklabels=label_map.keys())
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ==============================
# SAVE MODEL
# ==============================
MODEL_PATH = "alzheimer_resnet18_model.pth"
torch.save(model.state_dict(), MODEL_PATH)

print("\n==========================================")
print("PIPELINE COMPLETE")
print(f"Model saved at: {MODEL_PATH}")
print("==========================================")

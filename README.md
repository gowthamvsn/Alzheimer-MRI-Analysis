# Alzheimer’s MRI Classification using ResNet18

This project classifies brain MRI images into Alzheimer’s disease stages
using a deep learning model based on **ResNet18**.

The goal is to explore how convolutional neural networks can assist
in early detection of Alzheimer’s disease from MRI scans.

---

## Dataset
- Brain MRI images for Alzheimer’s disease classification
- Typical classes include:
  - Non-Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented

> ⚠️ Dataset is not included due to size and licensing constraints.

---

## Model
- Architecture: **ResNet18**
- Framework: PyTorch / TensorFlow (depending on implementation)
- Transfer learning with pretrained weights
- Final classification layer adapted for MRI classes

---


<img width="1796" height="460" alt="image" src="https://github.com/user-attachments/assets/4ce7465d-292e-4199-96e9-046bdf0d8202" />

## Workflow
1. Load and preprocess MRI images
2. Apply data normalization and augmentation
3. Train ResNet18-based classifier
4. Evaluate performance using accuracy and confusion matrix

---

## File
- `alzheimer_mri_resnet18.ipynb` – Complete training and evaluation pipeline

📌 **Start here:** `alzheimer_mri_resnet18.ipynb`

---

## Results
The trained model demonstrates strong performance in distinguishing
between Alzheimer’s disease stages based on MRI scans.

---

## Author
Gowtham

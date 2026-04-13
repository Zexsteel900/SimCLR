# Self-Supervised Learning using SimCLR on EuroSAT

## Overview
This project implements a self-supervised learning (SSL) framework using SimCLR to learn visual representations from unlabeled satellite images. The goal is to demonstrate that representations learned without labels can outperform supervised models when labeled data is limited.

The model is trained on the EuroSAT dataset and evaluated using linear probing, fine-tuning, and comparison with a supervised baseline.

---

## Dataset
**EuroSAT (RGB)**  
- 27,000 satellite images  
- Image size: 64 × 64  
- 10 land-cover classes:
  - AnnualCrop, Forest, HerbaceousVegetation, Highway
  - Industrial, Pasture, PermanentCrop, Residential
  - River, SeaLake

---

## Methodology

### SimCLR Framework
The implementation follows the SimCLR architecture:

- **Data Augmentation**
  - Random resized crop
  - Horizontal flip
  - Color jitter
  - Grayscale
  - Gaussian blur  
  → Generates two correlated views of the same image

- **Encoder**
  - ResNet-18 backbone
  - Final fully connected layer removed

- **Projection Head**
  - 2-layer MLP (512 → 256 → 128)

- **Loss Function**
  - NT-Xent (contrastive loss)

---

## Experiments

The model is evaluated under different label availability settings:

- **Baseline (Supervised from scratch)**
- **Linear Probe (Frozen encoder)**
- **Fine-tuning (Full model update)**

---

## Results

| Method        | 1% Labels | 5% Labels | 10% Labels |
|--------------|----------|----------|-----------|
| Baseline     | 25.19%   | 59.39%   | 72.41%    |
| Linear Probe | 80.94%   | 84.20%   | 84.94%    |
| Fine-tuning  | 78.46%   | 87.63%   | 88.98%    |

**Key Observation:**  
Self-supervised learning significantly outperforms supervised learning when labeled data is scarce.

---

## Project Structure

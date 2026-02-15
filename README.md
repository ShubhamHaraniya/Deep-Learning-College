# Stress-Testing CNNs: Baseline vs SE-Attention on CIFAR-10

**Deep Learning — Assignment 1**

**Group Members:** Vasishth Bhatt (M25CSA007), Shubham Haraniya (M25CSA013), Shivam Madhav Kenche (M25CSA028), Vidhan Savaliya (M25CSA031)

---

## Overview

This project compares a **Baseline CNN** against an **SE-Attention CNN** (Squeeze-and-Excitation) on the CIFAR-10 dataset. The pipeline trains both models, evaluates per-class accuracy, generates Grad-CAM visualizations, confusion matrices, and failure case analysis.

## Requirements

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** For GPU support, install PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

## Project Structure

```
├── main.py              # Complete training and evaluation pipeline
├── report.tex           # LaTeX report
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── data/                # CIFAR-10 dataset (auto-downloaded)
└── output/              # All generated outputs
    ├── models/          # Saved model weights (.pth)
    ├── plots/           # Training curves, accuracy comparison,
    │                    #   confusion matrices, Grad-CAM comparisons,
    │                    #   failure cases
    ├── gradcam/         # Individual Grad-CAM heatmaps
    │   ├── baseline/
    │   └── se_attention/
    └── results/         # Class-wise accuracy JSONs
```

## How to Run

### 1. Train and Evaluate (Full Pipeline)

```bash
python main.py
```

This single command runs the entire pipeline:
1. Downloads CIFAR-10 (if not already in `./data/`)
2. Trains the **Baseline CNN** for 15 epochs
3. Trains the **SE-Attention CNN** for 15 epochs
4. Evaluates both models and saves per-class accuracy JSONs
5. Generates training curves, accuracy bar chart, and confusion matrices
6. Produces Grad-CAM visualizations for cat, dog, bird, airplane, and ship
7. Identifies and visualizes failure cases

> **GPU:** The script automatically uses CUDA if available, otherwise falls back to CPU.

### 2. Reproduce Results (Re-run from Saved Models)

If trained models already exist in `output/models/`, the script **skips training** and directly loads them for evaluation. To retrain from scratch:

```bash
# Delete existing models to force retraining
rm output/models/baseline_model.pth
rm output/models/se_attention_model.pth
python main.py
```

### 3. Reproducibility

The script uses **Random Seed 42** for all random number generators (Python, NumPy, PyTorch, CUDA) and sets `torch.backends.cudnn.deterministic = True` to ensure reproducible results.

## Key Results

| Class | Baseline | SE-Attention | Change |
|-------|----------|-------------|--------|
| airplane | 86.90% | 85.60% | -1.30% |
| automobile | 91.10% | 91.70% | +0.60% |
| bird | 88.70% | 72.50% | -16.20% |
| **cat** | 62.00% | **77.60%** | **+15.60%** |
| **deer** | 74.50% | **87.50%** | **+13.00%** |
| **dog** | 50.80% | **63.00%** | **+12.20%** |
| frog | 82.10% | 78.00% | -4.10% |
| **horse** | 71.90% | **82.20%** | **+10.30%** |
| ship | 86.60% | 88.80% | +2.20% |
| truck | 74.10% | 82.90% | +8.80% |
| **Overall** | 76.87% | **80.98%** | **+4.11%** |

## Reference

- J. Hu, L. Shen, and G. Sun, "Squeeze-and-Excitation Networks," CVPR 2018.

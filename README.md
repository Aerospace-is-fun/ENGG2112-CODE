# ENGG2112-CODE
Final code for the ENGG2112 project (Cloudy chance of Rain)
# ENGG-2112-CODE
### A Cloudy Chance of Rain — Severe Weather Classification from Satellite Radar Imagery

**ENGG2112 Machine Learning Project | University of Sydney | 2026**

Adrian Wong · Jake Earls · Yat Him Leung · Phi Tran

---

## Overview

This repository contains the full implementation of a hybrid **CNN + Random Forest** pipeline for classifying severe weather events from [SEVIR](https://github.com/MIT-AI-Accelerator/eie-sevir) satellite radar imagery.

A **ResNet-18** backbone (pretrained on ImageNet, adapted to single-channel infrared input) is used as a spatial feature extractor. The resulting 512-dimensional embeddings are passed to a **Random Forest** classifier to predict one of eight severe weather categories:

| Class | Class |
|---|---|
| Thunderstorm Wind | Hail |
| Flash Flood | Flood |
| Funnel Cloud | Heavy Rain |
| Lightning | Tornado |

An optional **No Storm** class (derived from unlabelled SEVIR events) can be included for open-world evaluation.

---

## Results Summary

| Metric | CNN only | CNN → RF |
|---|---|---|
| Precision | 0.326 | 0.916 |
| Recall | 0.273 | 0.720 |
| F1 Score | 0.285 | 0.793 |
| AUC (macro) | 0.777 | 0.922 |
| Specificity | 0.921 | 0.987 |
| NPV | 0.908 | 0.966 |

**TSS = 0.707** (NS Excluded model), exceeding Holmes [2008] (0.700) and Hobson et al. [2012] (0.616).

---

## Repository Structure

```
ENGG-2112-CODE/
├── random_forest_CNN_training.py                  # Full CNN training pipeline (ResNet-18, two modes)
├── random_forest.py   # Load saved checkpoint → CNN eval + RF training & eval
└── README.md
```

### `random_forest_CNN_training.py`
End-to-end training script. Handles data loading, preprocessing, model training with early stopping, and test evaluation.

**Two training modes** — set `TRAINING_MODE` at the top of the file:
- `"without_no_storm"` — trains on labelled storm events only (recommended)
- `"with_no_storm"` — includes unlabelled events as a No Storm class

Outputs per run:
- `sevir_storm_classifier_{mode}.pt` — best model checkpoint
- `sevir_storm_classifier_{mode}_label_encoder.pkl` — label encoder
- `sevir_confusion_matrix_{mode}.png`
- `sevir_roc_auc_{mode}.png`
- `sevir_training_history_{mode}.png`

### `random_forest.py`
Loads a saved CNN checkpoint without retraining. Runs:
1. CNN test set evaluation (confusion matrix + ROC-AUC)
2. CNN feature extraction → Random Forest training and evaluation
3. RF ROC-AUC curves
4. Single-image inference example

---

## Setup

### Requirements

```bash
pip install torch torchvision h5py numpy matplotlib scikit-learn pandas seaborn joblib
```

> **GPU (optional):** For AMD GPU support via DirectML, also install `torch-directml`. The scripts auto-detect CUDA → DirectML → CPU in that order. All training was conducted on CPU for this project.

### Data

Download the SEVIR dataset from the [official repository](https://github.com/MIT-AI-Accelerator/eie-sevir):

```
data/
└── sevir/
    ├── CATALOG.csv
    └── vil/
        ├── SEVIR_VIL_STORMEVENTS_2018_0101_0630.h5
        ├── SEVIR_VIL_STORMEVENTS_2018_0701_1231.h5
        └── ...
```

Update `CONFIG["data_dir"]` and `CONFIG["catalog_path"]` in either script if your paths differ.

---

## Usage

### 1. Train the CNN

```bash
python random_forest_CNN_training.py
```

Edit `TRAINING_MODE` at the top of `random_forest_CNN_training.py` to switch between `"without_no_storm"` and `"with_no_storm"`.

### 2. Evaluate and train the Random Forest

```bash
python random_forest.py
```

This loads the saved checkpoint and label encoder, re-creates the same train/test split (seed 42), evaluates the CNN, trains the RF on CNN embeddings, and saves results.

> **Important:** `random_forest.py` expects `CHECKPOINT_PATH` and `LABEL_ENCODER_PATH` to point to files produced by `random_forest_CNN_training.py`. Update these paths at the top of the script if needed.

---

## Model Architecture

```
Input: 384×384 infrared image (ir069 or ir107 channel)
  ↓ Resize to 64×64
  ↓ ResNet-18 encoder (single-channel, ImageNet weights averaged)
  ↓ Global average pooling → 512-dim embedding
  ↓ FC(512→256) + ReLU + Dropout
  ↓ FC(256→num_classes)
  → CNN prediction

  OR

  512-dim embedding
  ↓ Random Forest (100 trees, max_depth=None, sqrt features, bootstrap)
  → RF prediction
```

### Training configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 5×10⁻³ |
| Weight decay | 1.5×10⁻⁴ |
| Batch size | 32 |
| LR schedule | 5-epoch warmup + cosine annealing (min 10⁻⁶) |
| Max epochs | 100 |
| Early stopping patience | 50 |
| Input size | 64×64 |
| Random seed | 42 |

---

## Citation

If you use this code, please cite the datasets:

- **SEVIR:** Veillette et al., *SEVIR: A Storm Event Imagery Dataset for Deep Learning Applications in Radar and Satellite Meteorology*, NeurIPS 2020.
- **SVRIMG:** Louisiana State University, *Mean storms: Composites of radar reflectivity images during two decades of severe thunderstorm events*, International Journal of Climatology, 2020.

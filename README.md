# Multi-Task Liquid and Bottle Classification

A comprehensive multi-task learning (MTL) framework for simultaneous liquid classification (19 classes) and bottle type identification (4 types) using sensor data from robotic applications.

## 📋 Project Overview

This project implements and compares multiple deep learning architectures for robotic liquid/bottle identification:

- **MTL Model**: CNN + Transformer with shared backbone and two independent classification heads
  - Liquid classification: 19 classes (water, ethanol, milk, juice, oil, sauces, sugars, syrup, vinegar, etc.)
  - Bottle classification: 4 types (pet01, pet02, glass01, glass02)
  
- **Baseline Models**: Single-task liquid classification for comparison
  - LSTM (Bidirectional LSTM with attention)
  - Transformer (Pure Transformer encoder)
  - XGBoost (Statistical feature extraction)

## 🔧 Requirements

```
torch>=2.0.0
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.3
seaborn>=0.11
plotly>=5.0
xgboost>=1.7
tqdm>=4.50
tensorboard>=2.0
```

### Installation

```bash
conda create -n mtl-liquid python=3.10
conda activate mtl-liquid
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 📁 Directory Structure

```
Robot_hand/
├── README.md                       # Documentation
├── requirements.txt                # Python dependencies
├── datasets_New/                   # Input data (19 liquid classes)
├── MLC/                            # Output directory (checkpoints & visualizations)
├── baseline_runs_*/                # Baseline training results
│
├── train_mtl.py                    # MTL model training
├── baseline_lstm.py                # LSTM baseline
├── baseline_transformer.py         # Transformer baseline
├── baseline_xgboost.py             # XGBoost baseline
├── CNN_transformer_mtl.py          # MTL architecture
├── dataloader_mtl.py               # Data loading utilities
├── mtl_visualize_pca_tsne.py       # Embedding extraction
├── visualize_embeddings.ipynb      # Interactive visualization
├── submit_mtl_train.sh             # SLURM: MTL training
└── submit_baselines.sh             # SLURM: Baseline training (12 configs)
```

## 📊 Input Data Format

**CSV files in `datasets_New/liq-*/`:**
- Columns: 4 sensor channels (mcap_1, mcap_2, mcap_3, mcap_4)
- Rows: 384 timesteps per sequence
- Labels: Liquid class (folder name), Bottle type (extracted from filename: pet01, pet02, glass01, glass02)

**Preprocessing:**
- Standardization (training set statistics)
- Middle 80% phase extraction
- Optional delta features (4-step difference)

## 🚀 Quick Start

### 1. MTL Training

```bash
python train_mtl.py \
    --csv_list datasets_New/dataset_split_balanced.csv \
    --allow_split random_by_class \
    --val_ratio 0.2 \
    --epochs 1000 \
    --batch_size 64 \
    --lr 5e-4 \
    --require_cuda
```

### 2. Baseline Training

```bash
# Single baseline
python baseline_lstm.py \
    --csv_list datasets_New/dataset_split_balanced.csv \
    --allow_split stratified_by_bid \
    --feature_cols mcap_delta_1-4 \
    --epochs 500 \
    --batch_size 32

# All 12 configurations (3 models × 2 splits × 2 features)
sbatch submit_baselines.sh
```

### 3. Visualization

```bash
# Extract embeddings
python mtl_visualize_pca_tsne.py \
    --checkpoint MLC/output_mtl_[timestamp]/best_mtl.pt \
    --csv_list datasets_New/dataset_split_balanced.csv

# Interactive dashboard
jupyter notebook visualize_embeddings.ipynb
```

## 📈 Model Architectures

**MTL Model (CNN + Transformer):**
- CNN encoder (4 layers) + Transformer (512d, 8 heads, 4 layers, RoPE)
- Dual heads: Liquid (19 classes) + Bottle (4 types)
- Loss: `2.2×L_liquid + 0.1×L_bottle`

**LSTM Baseline:** Bidirectional LSTM (512 hidden, 4 layers) + attention → 19 classes

**Transformer Baseline:** Pure Transformer encoder (512d, 8 heads, 4 layers) → 19 classes

**XGBoost Baseline:** Statistical features (8 stats × 4 channels = 32D) → gradient boosting

##  Analysis & Visualization

```bash
# Extract 2D embeddings from trained MTL model
python mtl_visualize_pca_tsne.py \
    --checkpoint MLC/output_mtl_[timestamp]/best_mtl.pt \
    --csv_list datasets_New/dataset_split_balanced.csv

# Interactive Jupyter dashboard
jupyter notebook visualize_embeddings.ipynb
```

**Outputs:** PCA/t-SNE projections, Plotly dashboards, K-means clustering (k=5), 300 DPI exports

## 🖥️ HPC Submission

```bash
# Submit MTL training
sbatch submit_mtl_train.sh

# Submit all 12 baseline configurations
sbatch submit_baselines.sh

# Monitor jobs
squeue -u $USER
```

**Resources:** GPU L40S (64GB), 12 CPU cores, 10-120 hours

## 📝 Output Structure

```
MLC/
├── output_mtl_[timestamp]/
│   ├── best_mtl.pt            # Best checkpoint
│   ├── config.json            # Training config
│   └── events.out.tfevents.*  # TensorBoard logs
│
└── visualize_mtl_[timestamp]/
    ├── embeddings.csv         # 2D coordinates + labels
    ├── pca_liquid.png
    ├── pca_bottle.png
    ├── tsne_liquid.png
    └── tsne_bottle.png
```

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not available | Run `python -c "import torch; print(torch.cuda.is_available())"` or use `--no_require_cuda` |
| Out of memory | Reduce `--batch_size` or use AMP (enabled by default) |
| Data loading errors | Verify CSV paths and file permissions: `ls -la datasets_New/` |

---

**Last Updated:** 2026-02-27 | **Python 3.10+** | **PyTorch 2.0+**

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
tqdm>=4.50
tensorboard>=2.0
```

### Installation

```bash
# Create conda environment
conda create -n mtl-liquid python=3.10
conda activate mtl-liquid

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pandas numpy scikit-learn matplotlib seaborn plotly tqdm tensorboard
```

## 📁 Directory Structure

```
Robot_hand/
├── README.md                          # This file
├── Requirements.txt                   # Python dependencies
│
├── datasets_New/                      # Input data directory
│   ├── dataset_index_all.csv         # Full dataset index
│   ├── dataset_split_balanced.csv    # Balanced split metadata
│   ├── dataset_split_day.csv         # Day-based split metadata
│   ├── liq-water/                    # Liquid class folders (19 total)
│   ├── liq-ethanol/
│   ├── liq-milk/
│   ├── liq-oil/
│   ├── liq-dishwash/
│   ├── liq-handwash/
│   ├── liq-grape_juice/
│   ├── liq-milkshake/
│   ├── liq-salt002/
│   ├── liq-salt004/
│   ├── liq-salt006/
│   ├── liq-salt008/
│   ├── liq-soy_sauce/
│   ├── liq-sugar005/
│   ├── liq-sugar010/
│   ├── liq-sugar015/
│   ├── liq-sugar020/
│   ├── liq-syrup/
│   └── liq-vinegar/
│
├── MLC/                               # Output directory for MTL training
│   ├── output_mtl_[timestamp]/        # Training checkpoints
│   └── visualize_mtl_[timestamp]/     # Embedding visualization outputs
│
├── baseline_runs_20260226_184024/    # Baseline training results
│   ├── xgboost_stratified_by_bid_mcap_delta_1-4/
│   ├── xgboost_stratified_by_bid_mcap_raw_1-4/
│   ├── ... (12 configurations total)
│   └── results_summary.txt
│
├── Core Training Scripts
│   ├── train_mtl.py                  # MTL model training script
│   ├── CNN_transformer_mtl.py         # MTL architecture definition
│   ├── dataloader_mtl.py              # Data loading and preprocessing
│   ├── baseline_lstm.py               # LSTM baseline training
│   ├── baseline_transformer.py        # Transformer baseline training
│   └── baseline_xgboost.py            # XGBoost baseline training
│
├── Visualization & Analysis
│   ├── mtl_visualize_pca_tsne.py     # PCA/t-SNE embedding extraction
│   ├── visualize_embeddings.ipynb    # Interactive embedding visualization
│   ├── tsne_visualization_hq.png     # High-quality t-SNE plot output
│   └── embeddings.csv                # Extracted embeddings data
│
└── SLURM Submission Scripts
    ├── submit_mtl_train.sh            # MTL training job submission
    └── submit_baselines.sh            # Baseline training jobs (12 configs)
```

## 📊 Input Data Format

Each liquid class folder contains CSV files with sensor data:

**CSV Structure:**
- **Columns**: 4 sensor channels (mcap_1, mcap_2, mcap_3, mcap_4)
- **Rows**: Time series samples (384 timesteps per sequence)
- **Preprocessing**: 
  - Standardization using training set statistics
  - Extraction of middle 80% phase of acquisition
  - Optional delta feature computation (4-step difference)
- **Labels**:
  - Liquid class: Folder name (e.g., liq-water)
  - Bottle type: Extracted from filename (pet01, pet02, glass01, glass02)

## 🚀 Quick Start

### 1. MTL Model Training

```bash
# Single GPU training
python train_mtl.py \
    --csv_list datasets_New/dataset_split_balanced.csv \
    --allow_split random_by_class \
    --val_ratio 0.2 \
    --epochs 1000 \
    --batch_size 64 \
    --lr 5e-4 \
    --seed 42 \
    --require_cuda

# Or submit to HPC with SLURM
sbatch submit_mtl_train.sh
```

**Key Arguments:**
- `--csv_list`: Path to dataset index CSV
- `--allow_split`: Data split strategy (random_by_class, stratified_by_bid, by_device)
- `--val_ratio`: Validation set ratio (0.0-1.0)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--require_cuda`: Enforce GPU usage

**Output:**
- Best model checkpoint: `MLC/output_mtl_[timestamp]/best_mtl.pt`
- Training logs: TensorBoard in `MLC/output_mtl_[timestamp]/`
- Per-class metrics on test set (accuracy, F1, confusion matrix)

### 2. Baseline Model Training

```bash
# Train single baseline
python baseline_lstm.py \
    --csv_list datasets_New/dataset_split_balanced.csv \
    --allow_split stratified_by_bid \
    --feature_cols mcap_delta_1-4 \
    --epochs 500 \
    --batch_size 32

# Or train all 12 baseline configurations at once
sbatch submit_baselines.sh
```

**Baseline Configurations (3 models × 2 splits × 2 features = 12 jobs):**
- **Models**: xgboost, lstm, transformer
- **Split modes**: stratified_by_bid, by_device
- **Features**: mcap_delta_1-4 (4-step difference), mcap_raw_1-4 (raw values)

### 3. Embedding Visualization

```bash
# Extract MTL embeddings
python mtl_visualize_pca_tsne.py \
    --checkpoint MLC/output_mtl_[timestamp]/best_mtl.pt \
    --csv_list datasets_New/dataset_split_balanced.csv \
    --output_dir MLC/visualize_mtl_[timestamp]/

# Interactive visualization in Jupyter
jupyter notebook visualize_embeddings.ipynb
```

## 📈 Model Architectures

### MTL Model (CNN + Transformer)

**Backbone:**
- CNN encoder: 4 conv layers → adaptive pooling
- Transformer encoder: 512 hidden dims, 8 attention heads, 4 layers, RoPE positional encoding

**Heads:**
- Liquid classification: Softmax over 19 classes
- Bottle classification: Softmax over 4 types

**Multi-task Loss:**
```
L_total = 2.2 × L_liquid + 0.1 × L_bottle
```

### LSTM Baseline

- Bidirectional LSTM: 512 hidden units, 4 layers
- Attention mechanism for sequence pooling
- Output: 19-class liquid classifier

### Transformer Baseline

- Pure Transformer encoder: 512 dims, 8 heads, 4 layers
- Global average pooling
- Output: 19-class liquid classifier

### XGBoost Baseline

- Statistical feature extraction (mean, std, min, max, percentiles)
- 32-dimensional feature vector (8 stats × 4 channels)
- Gradient boosting classifier

## 📊 Performance Results

### Best Models (Stratified-by-BID + Delta Features)

| Model | Accuracy | F1-Score |
|-------|----------|----------|
| **LSTM** | 91.56% | 0.9134 |
| XGBoost | 79.39% | 0.7892 |
| Transformer | 51.86% | 0.5043 |
| MTL | TBD | TBD |

**Key Findings:**
- Delta features significantly outperform raw features (15-30pp improvement)
- Stratified-by-bid split (preserving bottle ID distribution) > by_device split
- LSTM shows superior performance over Transformer and XGBoost

## 🔍 Analysis & Visualization

### Embedding Space Analysis

```bash
python mtl_visualize_pca_tsne.py  # Extract 2D embeddings via PCA/t-SNE
```

**Outputs:**
- `pca_liquid.png`: PCA projection of liquid classes
- `pca_bottle.png`: PCA projection of bottle types
- `tsne_liquid.png`: t-SNE projection of liquid classes (high-res)
- `tsne_bottle.png`: t-SNE projection of bottle types
- `embeddings.csv`: Full embedding coordinates with labels
- `summary.json`: Configuration metadata

### Interactive Dashboard

Run the Jupyter notebook:
```bash
jupyter notebook visualize_embeddings.ipynb
```

**Features:**
- 4 static matplotlib plots (PCA/t-SNE × liquid/bottle)
- 4 interactive Plotly dashboards with hover information
- K-means clustering analysis (k=5)
- Data statistics and class distribution
- High-quality PNG export (300 DPI)

## 🖥️ HPC Submission

### SLURM Configuration

**MTL Training:**
```bash
sbatch submit_mtl_train.sh
```

**Baseline Training (all 12 configs):**
```bash
sbatch submit_baselines.sh
```

**Monitor Jobs:**
```bash
squeue -u $USER
squeue -j <job_id> --long
```

**Resource Requirements (per job):**
- GPU: L40S (64GB VRAM)
- CPU: 12 cores
- Time: 10-120 hours (depending on epochs and model)
- Memory: ~80GB

## 📝 Output Structure

```
MLC/
├── output_mtl_20260227_073453/
│   ├── best_mtl.pt               # Best checkpoint
│   ├── final_mtl.pt              # Final checkpoint
│   ├── config.json               # Training config
│   ├── class_map.json            # Liquid class idx mapping
│   └── events.out.tfevents.*     # TensorBoard logs
│
└── visualize_mtl_20260227_073453/
    ├── embeddings.csv            # 8 columns: method, x, y, liquid_*, bottle_*, path
    ├── summary.json              # Metadata
    ├── pca_liquid.png
    ├── pca_bottle.png
    ├── tsne_liquid.png
    └── tsne_bottle.png
```

## 🔧 Troubleshooting

**CUDA not available:**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU-only (slower)
python train_mtl.py --no_require_cuda
```

**Memory issues:**
- Reduce `--batch_size`
- Enable gradient checkpointing: `--use_gradient_checkpointing`
- Use mixed precision: Already enabled with AMP

**Data loading errors:**
- Verify CSV paths in `--csv_list`
- Check file permissions: `ls -la datasets_New/`
- Validate CSV format: `head -5 datasets_New/liq-water/*.csv`

## 📚 Citation

```bibtex
@project{mtl_liquid_classification_2026,
  title={Multi-Task Learning for Robotic Liquid and Bottle Classification},
  author={Your Name},
  year={2026}
}
```

## 📞 Contact

For questions or issues, please open an issue or contact the maintainer.

---

**Last Updated:** 2026-02-27  
**Status**: Active Development  
**Python Version**: 3.10+  
**Framework**: PyTorch 2.0+

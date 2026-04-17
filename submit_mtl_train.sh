#!/bin/bash

# ============================================================
# Multi-Task Learning (MTL) Training Script
# Liquid Classification + Bottle Type Classification
# ============================================================

#SBATCH --job-name=mtl_train
#SBATCH --partition=gpuA
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=k09562zs@manchester.ac.uk

# Print job info
echo "=========================================="
echo "Job: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $SLURM_GPUS"
echo "Start time: $(date)"
echo "=========================================="

# Change to work directory
cd /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion
echo "Working directory: $(pwd)"

# Activate conda environment
echo ""
echo "Activating conda environment..."
source ~/.bashrc
conda activate cgtest

# Check environment
echo ""
echo "Environment info:"
echo "  Python: $(which python)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "  GPU Available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Create logs directory if not exists
mkdir -p logs

# Run multi-task training
echo ""
echo "Starting MTL training..."
echo "=========================================="

echo "stratified_by_bid + delta features"
python train_mtl.py \
  --data_root /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/datasets_New \
  --out_dir /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/MLC/output_mtl_$(date +%Y%m%d_%H%M%S) \
  --split_mode stratified_by_bid \
  --d_model 512 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --dropout 0.05 \
  --pooling_strategy multi \
  --use_residual \
  --batch_size 64 \
  --epochs 500 \
  --lr 5e-4 \
  --wd 1e-2 \
  --label_smoothing 0.0 \
  --grad_clip 1.0 \
  --liquid_weight 2.2 \
  --bottle_weight 0.1 \
  --train_ratio 0.7 \
  --val_ratio 0.3 \
  --seq_len 384 \
  --features "mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4" \
  --phase "all" \
  --patience 50 \
  --min_epochs 20 \
  --min_delta 1e-4 \
  --num_workers 8 \
  --require_cuda \
  --no_early_stop \
  --seed 256


echo "stratified_by_bid + raw features"
python train_mtl.py \
  --data_root /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/datasets_New \
  --out_dir /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/MLC/output_mtl_$(date +%Y%m%d_%H%M%S) \
  --split_mode stratified_by_bid \
  --d_model 512 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --dropout 0.05 \
  --pooling_strategy multi \
  --use_residual \
  --batch_size 64 \
  --epochs 500 \
  --lr 5e-4 \
  --wd 1e-2 \
  --label_smoothing 0.0 \
  --grad_clip 1.0 \
  --liquid_weight 2.2 \
  --bottle_weight 0.1 \
  --train_ratio 0.7 \
  --val_ratio 0.3 \
  --seq_len 384 \
  --features "mcap_raw_1,mcap_raw_2,mcap_raw_3,mcap_raw_4" \
  --phase "all" \
  --patience 50 \
  --min_epochs 20 \
  --min_delta 1e-4 \
  --num_workers 8 \
  --require_cuda \
  --no_early_stop \
  --seed 256

echo "by_device + delta features"
python train_mtl.py \
  --data_root /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/datasets_New \
  --out_dir /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/MLC/output_mtl_$(date +%Y%m%d_%H%M%S) \
  --split_mode by_device \
  --d_model 512 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --dropout 0.05 \
  --pooling_strategy multi \
  --use_residual \
  --batch_size 64 \
  --epochs 500 \
  --lr 5e-4 \
  --wd 1e-2 \
  --label_smoothing 0.0 \
  --grad_clip 1.0 \
  --liquid_weight 2.2 \
  --bottle_weight 0.1 \
  --train_ratio 0.7 \
  --val_ratio 0.3 \
  --seq_len 384 \
  --features "mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4" \
  --phase "all" \
  --patience 50 \
  --min_epochs 20 \
  --min_delta 1e-4 \
  --num_workers 8 \
  --require_cuda \
  --no_early_stop \
  --seed 256

echo "by_device + raw features"
python train_mtl.py \
  --data_root /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/datasets_New \
  --out_dir /mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Capacitive_Temporal_Fusion/MLC/output_mtl_$(date +%Y%m%d_%H%M%S) \
  --split_mode by_device \
  --d_model 512 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 1024 \
  --dropout 0.05 \
  --pooling_strategy multi \
  --use_residual \
  --batch_size 64 \
  --epochs 500 \
  --lr 5e-4 \
  --wd 1e-2 \
  --label_smoothing 0.0 \
  --grad_clip 1.0 \
  --liquid_weight 2.2 \
  --bottle_weight 0.1 \
  --train_ratio 0.7 \
  --val_ratio 0.3 \
  --seq_len 384 \
  --features "mcap_raw_1,mcap_raw_2,mcap_raw_3,mcap_raw_4" \
  --phase "all" \
  --patience 50 \
  --min_epochs 20 \
  --min_delta 1e-4 \
  --num_workers 8 \
  --require_cuda \
  --no_early_stop \
  --seed 256


# Job completion
echo ""
echo "=========================================="
echo "End time: $(date)"
echo "Status: $?"
echo "=========================================="

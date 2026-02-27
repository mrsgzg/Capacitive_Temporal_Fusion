#!/bin/bash
#SBATCH --job-name=baseline_comparison
#SBATCH --output=baseline_comparison_%J.log
#SBATCH --time=48:00:00
#SBATCH --partition=gpuL
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12

# Activate conda
source ~/.bashrc
conda activate cgtest

cd "$SLURM_SUBMIT_DIR"

echo "=========================================="
echo "Starting Baseline Comparison Experiments"
echo "========== 12 Total Configurations ========="
echo "Job ID: $SLURM_JOB_ID"
echo "Submitted: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

timestamp=$(date +%Y%m%d_%H%M%S)
base_out_dir="baseline_runs_${timestamp}"
mkdir -p "$base_out_dir"

# Define configuration variants
declare -a split_modes=("stratified_by_bid" "by_device")
declare -a feature_variants=(
    "mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4:delta"
    "mcap_raw_1,mcap_raw_2,mcap_raw_3,mcap_raw_4:raw"
)

# Configuration counter
config_num=0
total_configs=12

# Function to run a baseline model
run_baseline() {
    local model=$1
    local split_mode=$2
    local feature_cols=$3
    local feat_label=$4
    local config_num=$5
    
    echo ""
    echo "=========================================="
    echo "Config $config_num: $model | Split: $split_mode | Features: $feat_label"
    echo "=========================================="
    echo "Start time: $(date)"
    
    out_dir="${base_out_dir}/baseline_${model}_${split_mode}_${feat_label}_${timestamp}"
    mkdir -p "$out_dir"
    
    case $model in
        xgboost)
            python3 baseline_xgboost.py \
                --data_root datasets_New \
                --feature_cols "$feature_cols" \
                --phase_keep true \
                --xgb_max_depth 6 \
                --xgb_lr 0.1 \
                --xgb_subsample 0.8 \
                --xgb_colsample 0.8 \
                --xgb_early_stop 50 \
                --epochs 500 \
                --seed 256 \
                --split_mode "$split_mode" \
                --train_ratio 0.7 \
                --out_dir "$out_dir" 2>&1 | tee "${out_dir}/training.log"
            ;;
        lstm)
            python3 baseline_lstm.py \
                --data_root datasets_New \
                --feature_cols "$feature_cols" \
                --seq_len 384 \
                --phase_keep true \
                --lstm_hidden 512 \
                --lstm_layers 4 \
                --dropout 0.05 \
                --epochs 500 \
                --batch_size 64 \
                --lr 5e-4 \
                --weight_decay 1e-2 \
                --grad_clip 1.0 \
                --num_workers 8 \
                --seed 256 \
                --split_mode "$split_mode" \
                --train_ratio 0.7 \
                --out_dir "$out_dir" 2>&1 | tee "${out_dir}/training.log"
            ;;
        transformer)
            python3 baseline_transformer.py \
                --data_root datasets_New \
                --feature_cols "$feature_cols" \
                --seq_len 384 \
                --phase_keep true \
                --d_model 512 \
                --nhead 8 \
                --num_layers 4 \
                --dim_ff 1024 \
                --dropout 0.05 \
                --epochs 500 \
                --batch_size 64 \
                --lr 5e-4 \
                --weight_decay 1e-2 \
                --grad_clip 1.0 \
                --num_workers 8 \
                --seed 256 \
                --split_mode "$split_mode" \
                --train_ratio 0.7 \
                --out_dir "$out_dir" 2>&1 | tee "${out_dir}/training.log"
            ;;
    esac
    
    status=$?
    echo "Completed with status: $status"
    echo "End time: $(date)"
    echo ""
    
    return $status
}

# Run all configurations
all_results=""

for model in xgboost lstm transformer; do
    for split_mode in "${split_modes[@]}"; do
        for feature_variant in "${feature_variants[@]}"; do
            IFS=':' read -r feature_cols feat_label <<< "$feature_variant"
            
            ((config_num++))
            echo " "
            echo ">>> Configuration $config_num / $total_configs <<<"
            echo " "
            
            run_baseline "$model" "$split_mode" "$feature_cols" "$feat_label" "$config_num"
            config_status=$?
            
            # Record result
            all_results="${all_results}${model} | ${split_mode} | ${feat_label}: Status $config_status\n"
        done
    done
done

# ========== Summary ==========
echo ""
echo "=========================================="
echo "BASELINE COMPARISON SUMMARY"
echo "=========================================="
echo "Timestamp: $timestamp"
echo "Total Configurations: $total_configs"
echo ""
echo "Configuration Matrix:"
echo "  Models: 3 (XGBoost, LSTM, Transformer)"
echo "  Split modes: 2 (stratified_by_bid, by_device)"
echo "  Feature types: 2 (mcap_delta, mcap_raw)"
echo ""

echo "========== Results Summary =========="
echo ""

for model in xgboost lstm transformer; do
    echo "=== $model ==="
    for split_mode in "${split_modes[@]}"; do
        for feature_variant in "${feature_variants[@]}"; do
            IFS=':' read -r feature_cols feat_label <<< "$feature_variant"
            out_dir="${base_out_dir}/baseline_${model}_${split_mode}_${feat_label}_${timestamp}"
            
            if [ -f "$out_dir/config.json" ]; then
                accuracy=$(grep -o '"accuracy": [0-9.]*' "$out_dir/config.json" | head -1 | cut -d' ' -f2)
                f1_score=$(grep -o '"f1_score": [0-9.]*' "$out_dir/config.json" | head -1 | cut -d' ' -f2)
                echo "  $split_mode | $feat_label: acc=$accuracy, f1=$f1_score"
            else
                echo "  $split_mode | $feat_label: Results not yet available"
            fi
        done
    done
    echo ""
done

echo ""
echo "Job completed at: $(date)"
echo "=========================================="

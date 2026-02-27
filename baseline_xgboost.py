# -*- coding: utf-8 -*-
"""
XGBoost Baseline for Liquid Classification
Extracts statistical features from time series and trains XGBoost
"""

import os
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataloader_mtl import (
    parse_csv_list, list_class_folders, select_classes,
    extract_bid_from_path, validate_csv, load_one_csv_as_sample,
    compute_train_scaler, make_splits_stratified_by_bid,
    make_splits_random_by_class, make_splits_by_device_id,
)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)


def build_scaler_samples(items, feature_cols, allow_missing_cols, phase_keep):
    samples = []
    for path, _ in items:
        try:
            df = pd.read_csv(path)
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols and not allow_missing_cols:
                continue
            cols_to_use = [c for c in feature_cols if c in df.columns]
            data = df[cols_to_use].values.astype(np.float32)
            if data.shape[1] < len(feature_cols):
                padded = np.zeros((data.shape[0], len(feature_cols)), dtype=np.float32)
                padded[:, :data.shape[1]] = data
                data = padded
            if phase_keep:
                start = int(len(data) * 0.1)
                end = int(len(data) * 0.9)
                data = data[start:end]
            if data.shape[0] >= 2:
                samples.append(data)
        except Exception:
            continue
    if not samples:
        raise RuntimeError("No valid samples for scaler")
    return samples


def extract_statistical_features(x: np.ndarray) -> np.ndarray:
    """Extract statistical features from time series
    
    Args:
        x: (T, C) time series array
    
    Returns:
        (N_FEATURES,) feature vector
    """
    features = []
    for c in range(x.shape[1]):
        series = x[:, c]
        features.extend([
            np.mean(series),
            np.std(series),
            np.min(series),
            np.max(series),
            np.median(series),
            np.percentile(series, 25),
            np.percentile(series, 75),
            np.max(series) - np.min(series),  # range
        ])
    return np.array(features, dtype=np.float32)


def load_data_for_xgboost(items, scaler, phase_keep, feature_cols, allow_missing_cols):
    """Load data and extract features for XGBoost
    
    Args:
        items: list of (path, label_idx) tuples
        
    Returns:
        X: (N, N_FEATURES) feature matrix
        y: (N,) label vector
    """
    X_list = []
    y_list = []
    
    for path, label_idx in tqdm(items, desc="Loading data"):
        try:
            # Load CSV
            df = pd.read_csv(path)
            
            # Validate and preprocess
            if df.shape[0] == 0:
                print(f"[WARN] Empty CSV: {path}")
                continue
            
            # Keep only feature columns
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols and not allow_missing_cols:
                print(f"[WARN] Missing columns {missing_cols}: {path}")
                continue
            
            # Select available columns
            cols_to_use = [c for c in feature_cols if c in df.columns]
            data = df[cols_to_use].values.astype(np.float32)
            
            # Fill missing with zeros
            if data.shape[1] < len(feature_cols):
                padded = np.zeros((data.shape[0], len(feature_cols)), dtype=np.float32)
                padded[:, :data.shape[1]] = data
                data = padded
            
            # Select phases
            if phase_keep:
                # Simple phase selection - take middle 80% of data
                start = int(len(data) * 0.1)
                end = int(len(data) * 0.9)
                data = data[start:end]
            
            # Normalize using scaler
            if scaler is not None:
                data = scaler.transform(data)
            
            # Extract features
            features = extract_statistical_features(data)
            X_list.append(features)
            y_list.append(label_idx)
            
        except Exception as e:
            print(f"[ERROR] Failed to load {path}: {e}")
            continue
    
    if not X_list:
        raise RuntimeError("No valid data loaded")
    
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int32)
    
    return X, y


def train_xgboost(X_train, y_train, X_val, y_val, num_classes, args):
    """Train XGBoost model
    
    Returns:
        model: trained XGBoost Classifier
        history: list of validation accuracies per round
    """
    # Handle multi-class
    params = {
        "objective": "multi:softmax",
        "num_class": num_classes,
        "max_depth": args.xgb_max_depth,
        "learning_rate": args.xgb_lr,
        "subsample": args.xgb_subsample,
        "colsample_bytree": args.xgb_colsample,
        "random_state": args.seed,
        "eval_metric": "mlogloss",
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    evals = [(dtrain, "train"), (dval, "validation")]
    evals_result = {}
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.epochs,
        evals=evals,
        evals_result=evals_result,
        verbose_eval=10,
        early_stopping_rounds=args.xgb_early_stop,
    )
    
    return model, evals_result


def evaluate_xgboost(model, X, y, idx_to_name):
    """Evaluate XGBoost model"""
    preds = model.predict(xgb.DMatrix(X)).astype(int)
    acc = accuracy_score(y, preds)
    f1_mac = f1_score(y, preds, average="macro", zero_division=0)
    
    report = classification_report(y, preds, 
                                   target_names=list(idx_to_name.values()),
                                   digits=4, zero_division=0)
    cm = confusion_matrix(y, preds)
    
    return {
        "acc": acc,
        "f1": f1_mac,
        "report": report,
        "cm": cm,
        "preds": preds,
    }


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_root", type=str, default="datasets_New",
                        help="Path to data root")
    parser.add_argument("--feature_cols", type=str, default="mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4",
                        help="Comma-separated feature column names")
    parser.add_argument("--phase_keep", type=lambda x: x.lower() == "true", default=True,
                        help="Keep only center phases of data")
    parser.add_argument("--allow_missing_cols", type=lambda x: x.lower() == "true", default=False,
                        help="Allow missing columns (pad with zeros)")
    parser.add_argument("--seed", type=int, default=256, help="Random seed")
    parser.add_argument("--split_mode", type=str, default="stratified_by_bid",
                        choices=["random", "stratified_by_bid", "by_device"],
                        help="Data splitting strategy")
    parser.add_argument("--train_devices", type=str, default="d1,d2,d3",
                        help="Train devices for split_mode=by_device")
    parser.add_argument("--val_devices", type=str, default="d4",
                        help="Val devices for split_mode=by_device")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val_ratio", type=float, default=0.3, help="Val ratio")
    
    # Model
    parser.add_argument("--xgb_max_depth", type=int, default=6, help="XGBoost max depth")
    parser.add_argument("--xgb_lr", type=float, default=0.1, help="XGBoost learning rate")
    parser.add_argument("--xgb_subsample", type=float, default=0.8, help="XGBoost subsample")
    parser.add_argument("--xgb_colsample", type=float, default=0.8, help="XGBoost colsample")
    parser.add_argument("--xgb_early_stop", type=int, default=50, help="XGBoost early stop rounds")
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000, help="Number of boosting rounds")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers")
    
    # Output
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Create output directory
    if args.out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"baseline_xgboost_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"[INFO] Output directory: {args.out_dir}")
    
    # Parse feature columns
    feature_cols = [c.strip() for c in args.feature_cols.split(",")]
    print(f"[INFO] Feature columns: {feature_cols}")
    
    # Load class structure
    data_root = args.data_root
    all_classes = list_class_folders(data_root, prefix="liq-")
    print(f"[INFO] Found {len(all_classes)} liquid classes: {all_classes}")
    
    class_map = {c: i for i, c in enumerate(all_classes)}
    idx_to_name = {i: c for c, i in class_map.items()}
    num_liquid_classes = len(all_classes)
    
    # Collect data
    print("[INFO] Collecting data...")
    phase_keep_set = {"precontact", "closing", "hold"}
    class_to_files = {class_map[c]: [] for c in all_classes}
    for cls_name in all_classes:
        cls_path = os.path.join(data_root, cls_name)
        csvs = glob.glob(os.path.join(cls_path, "**", "*.csv"), recursive=True)
        for csv_file in csvs:
            if validate_csv(csv_file, phase_keep_set, feature_cols, args.allow_missing_cols)[0]:
                class_to_files[class_map[cls_name]].append(csv_file)
    
    total_samples = sum(len(v) for v in class_to_files.values())
    print(f"[INFO] Total samples: {total_samples}")
    
    # Split data (with different strategies)
    if args.split_mode == "stratified_by_bid":
        train_items, val_items = make_splits_stratified_by_bid(
            class_to_files, args.seed, args.train_ratio, args.val_ratio
        )
    elif args.split_mode == "by_device":
        train_items, val_items = make_splits_by_device_id(
            class_to_files, args.seed,
            train_devices=parse_csv_list(args.train_devices),
            val_devices=parse_csv_list(args.val_devices)
        )
    else:  # random
        train_items, val_items = make_splits_random_by_class(
            class_to_files, args.seed, args.train_ratio, args.val_ratio
        )
    
    print(f"[INFO] Split mode: {args.split_mode}")
    print(f"[INFO] Train: {len(train_items)}, Val: {len(val_items)}")
    
    # Compute scaler from training data
    print("[INFO] Computing scaler...")
    scaler_samples = build_scaler_samples(
        train_items, feature_cols, args.allow_missing_cols, args.phase_keep
    )
    scaler = compute_train_scaler(scaler_samples)
    
    # Load data
    print("[INFO] Loading and extracting features...")
    X_train, y_train = load_data_for_xgboost(
        train_items, scaler, args.phase_keep, feature_cols, args.allow_missing_cols
    )
    X_val, y_val = load_data_for_xgboost(
        val_items, scaler, args.phase_keep, feature_cols, args.allow_missing_cols
    )
    
    print(f"[INFO] X_train shape: {X_train.shape}, X_val shape: {X_val.shape}")
    
    # Train model
    print("[INFO] Training XGBoost...")
    model, evals_result = train_xgboost(X_train, y_train, X_val, y_val, num_liquid_classes, args)
    
    # Evaluate
    print("[INFO] Evaluating on validation set...")
    val_result = evaluate_xgboost(model, X_val, y_val, idx_to_name)
    
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)
    print(f"Accuracy: {val_result['acc']:.4f}")
    print(f"F1-Score: {val_result['f1']:.4f}")
    print("\nClassification Report:")
    print(val_result["report"])
    print("\nConfusion Matrix:")
    print(val_result["cm"])
    
    # Save results
    results_file = os.path.join(args.out_dir, "results.txt")
    with open(results_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("XGBOOST BASELINE - LIQUID CLASSIFICATION\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {val_result['acc']:.4f}\n")
        f.write(f"F1-Score: {val_result['f1']:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(val_result["report"])
        f.write("\nConfusion Matrix:\n")
        f.write(str(val_result["cm"]))
    
    # Save config
    config = {
        "model": "xgboost",
        "num_classes": num_liquid_classes,
        "classes": all_classes,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "accuracy": float(val_result["acc"]),
        "f1_score": float(val_result["f1"]),
        "args": vars(args),
    }
    
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[INFO] Results saved to {args.out_dir}")


if __name__ == "__main__":
    import glob
    import torch
    main()

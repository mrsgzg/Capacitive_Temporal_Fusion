# -*- coding: utf-8 -*-
"""
Multi-task training script for liquid + bottle classification
Trains CNN+Transformer model with two classification heads
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
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataloader_mtl import (
    parse_csv_list, list_class_folders, list_csvs, select_classes,
    extract_bid_from_path, extract_device_id, validate_csv, load_one_csv_as_sample,
    compute_train_scaler, create_dataloaders_mtl,
    make_splits_random_by_class, make_splits_stratified_by_bid, make_splits_by_device_id,
)
from CNN_transformer_mtl import CNNTransformerMTL


# ================== Utility Functions ==================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_cuda_env(require_cuda: bool):
    """Print CUDA environment info"""
    print("[ENV] torch:", torch.__version__)
    print("[ENV] torch.version.cuda:", torch.version.cuda)
    print("[ENV] cuda_available:", torch.cuda.is_available())
    if require_cuda and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA not available (--require_cuda enabled)")
    if torch.cuda.is_available():
        print("[ENV] gpu_name:", torch.cuda.get_device_name(0))
        print("[ENV] capability:", torch.cuda.get_device_capability(0))
        print("[ENV] cudnn:", torch.backends.cudnn.version())


def safe_torch_save(state: dict, path: str, retries: int = 5, sleep_s: float = 0.5):
    """Safer checkpoint save (atomic write)"""
    out_dir = os.path.dirname(path)
    os.makedirs(out_dir, exist_ok=True)
    tmp_path = path + ".tmp"

    last_err = None
    for _ in range(retries):
        try:
            torch.save(state, tmp_path)
            os.replace(tmp_path, path)
            return
        except Exception as e:
            last_err = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to save checkpoint: {path} -> {last_err}")


# ================== Evaluation ==================

@torch.no_grad()
def evaluate_mtl(model, loader, device, return_predictions=False):
    """Evaluate multi-task model
    
    Args:
        return_predictions: if True, return (y_liquid, p_liquid, y_bottle, p_bottle) 
                           instead of metrics dict
    """
    model.eval()
    y_liquid_list, p_liquid_list = [], []
    y_bottle_list, p_bottle_list = [], []
    
    for x, y_liquid, y_bottle, _ in loader:
        x = x.to(device, non_blocking=True)
        liquid_logits, bottle_logits = model(x)
        
        p_liquid = torch.argmax(liquid_logits, dim=1).cpu().numpy()
        p_bottle = torch.argmax(bottle_logits, dim=1).cpu().numpy()
        
        y_liquid_list.append(y_liquid.numpy())
        p_liquid_list.append(p_liquid)
        y_bottle_list.append(y_bottle.numpy())
        p_bottle_list.append(p_bottle)
    
    y_liquid = np.concatenate(y_liquid_list)
    p_liquid = np.concatenate(p_liquid_list)
    y_bottle = np.concatenate(y_bottle_list)
    p_bottle = np.concatenate(p_bottle_list)
    
    if return_predictions:
        return y_liquid, p_liquid, y_bottle, p_bottle
    
    return {
        "liquid_acc": float(accuracy_score(y_liquid, p_liquid)),
        "liquid_f1_macro": float(f1_score(y_liquid, p_liquid, average="macro", zero_division=0)),
        "liquid_f1_weighted": float(f1_score(y_liquid, p_liquid, average="weighted", zero_division=0)),
        "liquid_f1_micro": float(f1_score(y_liquid, p_liquid, average="micro", zero_division=0)),
        "bottle_acc": float(accuracy_score(y_bottle, p_bottle)),
        "bottle_f1_macro": float(f1_score(y_bottle, p_bottle, average="macro", zero_division=0)),
        "bottle_f1_weighted": float(f1_score(y_bottle, p_bottle, average="weighted", zero_division=0)),
        "bottle_f1_micro": float(f1_score(y_bottle, p_bottle, average="micro", zero_division=0)),
    }


# ================== Training ==================

def train_mtl_model(args, train_items, val_items, class_map, bid_map, phase_keep, 
                    feature_cols, scaler, idx_to_name, idx_to_bid, num_liquid_classes, num_bottle_classes):
    """Train multi-task model"""
    
    set_seed(args.seed)
    print_cuda_env(args.require_cuda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] device =", device)
    
    # Create dataloaders
    loader_train, loader_val, _, _ = create_dataloaders_mtl(
        train_items, val_items, args.seq_len, args.batch_size, args.num_workers,
        scaler, phase_keep, feature_cols, args.allow_missing_cols
    )
    
    # Create model
    in_ch = len(feature_cols)
    model = CNNTransformerMTL(
        seq_len=args.seq_len,
        in_ch=in_ch,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        num_liquid_classes=num_liquid_classes,
        num_bottle_classes=num_bottle_classes,
        pooling_strategy=args.pooling_strategy,
        use_residual=args.use_residual,
    ).to(device)
    
    if args.require_cuda:
        assert next(model.parameters()).is_cuda, "Model NOT on CUDA"
    
    print(f"[INFO] Model: CNNTransformerMTL")
    print(f"[INFO] Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[INFO] Liquid classes: {num_liquid_classes}, Bottle classes: {num_bottle_classes}")
    
    # Loss functions
    criterion_liquid = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    criterion_bottle = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = args.epochs * max(1, len(loader_train))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    use_amp = (device.type == "cuda") and (not args.no_amp)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    # TensorBoard
    tb_writer = SummaryWriter(os.path.join(args.out_dir, "runs", "mtl"))
    
    # Training loop
    best_score = -1e9
    best_path = os.path.join(args.out_dir, f"best_mtl.pt")
    history = []
    bad_epochs = 0
    global_step = 0
    
    print("\n[INFO] Starting multi-task training...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses_liquid, losses_bottle, losses_total = [], [], []
        
        for x, y_liquid, y_bottle, _ in loader_train:
            x = x.to(device, non_blocking=True)
            y_liquid = y_liquid.to(device, non_blocking=True)
            y_bottle = y_bottle.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                liquid_logits, bottle_logits = model(x)
                loss_liquid = criterion_liquid(liquid_logits, y_liquid)
                loss_bottle = criterion_bottle(bottle_logits, y_bottle)
                
                # Weighted sum of losses (1:1 by default)
                loss_total = args.liquid_weight * loss_liquid + args.bottle_weight * loss_bottle
            
            grad_scaler.scale(loss_total).backward()
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            
            losses_liquid.append(loss_liquid.item())
            losses_bottle.append(loss_bottle.item())
            losses_total.append(loss_total.item())
            
            global_step += 1
            tb_writer.add_scalar("loss/liquid", loss_liquid.item(), global_step)
            tb_writer.add_scalar("loss/bottle", loss_bottle.item(), global_step)
            tb_writer.add_scalar("loss/total", loss_total.item(), global_step)
        
        train_loss_liquid = float(np.mean(losses_liquid)) if losses_liquid else float("nan")
        train_loss_bottle = float(np.mean(losses_bottle)) if losses_bottle else float("nan")
        train_loss_total = float(np.mean(losses_total)) if losses_total else float("nan")
        
        val_metrics = evaluate_mtl(model, loader_val, device)
        
        # Compute combined score (average of liquid and bottle accuracy)
        combined_score = (val_metrics["liquid_acc"] + val_metrics["bottle_acc"]) / 2
        
        rec = {
            "epoch": epoch,
            "train_loss_liquid": train_loss_liquid,
            "train_loss_bottle": train_loss_bottle,
            "train_loss_total": train_loss_total,
            "val_liquid_acc": val_metrics["liquid_acc"],
            "val_liquid_f1_weighted": val_metrics["liquid_f1_weighted"],
            "val_bottle_acc": val_metrics["bottle_acc"],
            "val_bottle_f1_weighted": val_metrics["bottle_f1_weighted"],
            "combined_score": combined_score,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(rec)
        
        # TensorBoard logging
        tb_writer.add_scalar("metrics/val_liquid_acc", val_metrics["liquid_acc"], epoch)
        tb_writer.add_scalar("metrics/val_liquid_f1_weighted", val_metrics["liquid_f1_weighted"], epoch)
        tb_writer.add_scalar("metrics/val_liquid_f1_macro", val_metrics["liquid_f1_macro"], epoch)
        tb_writer.add_scalar("metrics/val_bottle_acc", val_metrics["bottle_acc"], epoch)
        tb_writer.add_scalar("metrics/val_bottle_f1_weighted", val_metrics["bottle_f1_weighted"], epoch)
        tb_writer.add_scalar("metrics/val_bottle_f1_macro", val_metrics["bottle_f1_macro"], epoch)
        tb_writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        
        print(f"[E{epoch:03d}] "
              f"L_l={train_loss_liquid:.4f} L_b={train_loss_bottle:.4f} "
              f"| liquid_acc={val_metrics['liquid_acc']:.4f} f1_w={val_metrics['liquid_f1_weighted']:.4f} "
              f"bottle_acc={val_metrics['bottle_acc']:.4f} f1_w={val_metrics['bottle_f1_weighted']:.4f} "
              f"| lr={rec['lr']:.2e}")
        
        improved = (combined_score >= best_score + args.min_delta)
        if improved:
            best_score = combined_score
            bad_epochs = 0
            safe_torch_save({
                "model": model.state_dict(),
                "args": vars(args),
                "class_map": class_map,
                "bid_map": bid_map,
            }, best_path)
        else:
            if (not args.no_early_stop) and (epoch >= args.min_epochs):
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[INFO] Early stop after {epoch} epochs")
                    break
    
    tb_writer.close()
    
    # Save history
    with open(os.path.join(args.out_dir, f"history_mtl.json"), "w") as f:
        json.dump(history, f, indent=2)
    
    return best_path, history


# ================== Testing ==================

@torch.no_grad()
def test_mtl_model(best_path, train_items, val_items, device, class_map, bid_map, 
                   args, idx_to_name, idx_to_bid, num_liquid_classes, num_bottle_classes, 
                   phase_keep, feature_cols, scaler):
    """Test multi-task model and output detailed metrics"""
    ckpt = torch.load(best_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    
    # Reconstruct model from checkpoint args
    in_ch = len(feature_cols)
    model = CNNTransformerMTL(
        seq_len=ckpt_args.get("seq_len", args.seq_len),
        in_ch=in_ch,
        d_model=ckpt_args.get("d_model", args.d_model),
        nhead=ckpt_args.get("nhead", args.nhead),
        num_layers=ckpt_args.get("num_layers", args.num_layers),
        dim_ff=ckpt_args.get("dim_ff", args.dim_ff),
        dropout=ckpt_args.get("dropout", args.dropout),
        num_liquid_classes=num_liquid_classes,
        num_bottle_classes=num_bottle_classes,
        pooling_strategy=ckpt_args.get("pooling_strategy", args.pooling_strategy),
        use_residual=ckpt_args.get("use_residual", args.use_residual),
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    print("[INFO] Testing best model...")
    _, loader_val, _, _ = create_dataloaders_mtl(
        train_items, val_items, args.seq_len, args.batch_size, args.num_workers,
        scaler, phase_keep, feature_cols, args.allow_missing_cols
    )
    
    # Get predictions
    y_liquid, p_liquid, y_bottle, p_bottle = evaluate_mtl(
        model, loader_val, device, return_predictions=True
    )
    
    # Calculate metrics
    liquid_acc = accuracy_score(y_liquid, p_liquid)
    liquid_f1_macro = f1_score(y_liquid, p_liquid, average="macro", zero_division=0)
    liquid_f1_weighted = f1_score(y_liquid, p_liquid, average="weighted", zero_division=0)
    liquid_f1_micro = f1_score(y_liquid, p_liquid, average="micro", zero_division=0)
    bottle_acc = accuracy_score(y_bottle, p_bottle)
    bottle_f1_macro = f1_score(y_bottle, p_bottle, average="macro", zero_division=0)
    bottle_f1_weighted = f1_score(y_bottle, p_bottle, average="weighted", zero_division=0)
    bottle_f1_micro = f1_score(y_bottle, p_bottle, average="micro", zero_division=0)
    
    # Generate reports
    liquid_report = classification_report(y_liquid, p_liquid, 
                                          target_names=list(idx_to_name.values()),
                                          digits=4, zero_division=0)
    bottle_report = classification_report(y_bottle, p_bottle,
                                          target_names=list(idx_to_bid.values()),
                                          digits=4, zero_division=0)
    
    # Confusion matrices
    liquid_cm = confusion_matrix(y_liquid, p_liquid)
    bottle_cm = confusion_matrix(y_bottle, p_bottle)
    
    # Print results
    print("\n" + "="*80)
    print("LIQUID CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Overall Accuracy:     {liquid_acc:.4f}")
    print(f"Macro F1-Score:       {liquid_f1_macro:.4f}  (equal weight per class)")
    print(f"Weighted F1-Score:    {liquid_f1_weighted:.4f}  (inverse class freq weighting)")
    print(f"Micro F1-Score:       {liquid_f1_micro:.4f}  (should equal accuracy)")
    print("\nClassification Report:")
    print(liquid_report)
    print("Confusion Matrix:")
    print(liquid_cm)
    
    print("\n" + "="*80)
    print("BOTTLE TYPE CLASSIFICATION RESULTS")
    print("="*80)
    print(f"Overall Accuracy:     {bottle_acc:.4f}")
    print(f"Macro F1-Score:       {bottle_f1_macro:.4f}  (equal weight per class)")
    print(f"Weighted F1-Score:    {bottle_f1_weighted:.4f}  (inverse class freq weighting)")
    print(f"Micro F1-Score:       {bottle_f1_micro:.4f}  (should equal accuracy)")
    print("\nClassification Report:")
    print(bottle_report)
    print("Confusion Matrix:")
    print(bottle_cm)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nLiquid Classification:")
    print(f"  Accuracy:       {liquid_acc:.4f}")
    print(f"  Macro F1:       {liquid_f1_macro:.4f}")
    print(f"  Weighted F1:    {liquid_f1_weighted:.4f}")
    print(f"  Micro F1:       {liquid_f1_micro:.4f}")
    print(f"\nBottle Classification:")
    print(f"  Accuracy:       {bottle_acc:.4f}")
    print(f"  Macro F1:       {bottle_f1_macro:.4f}")
    print(f"  Weighted F1:    {bottle_f1_weighted:.4f}")
    print(f"  Micro F1:       {bottle_f1_micro:.4f}")
    print(f"\nCombined Score: {(liquid_acc + bottle_acc) / 2:.4f}")
    print("\nNote: If Macro F1 ≈ Accuracy, classes are well-balanced.")
    print("      If Weighted F1 << Macro F1, there's significant class imbalance.")
    
    # Save results to file
    results_file = os.path.join(args.out_dir, "test_results.txt")
    with open(results_file, "w") as f:
        f.write("="*80 + "\n")
        f.write("LIQUID CLASSIFICATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy:  {liquid_acc:.4f}\n")
        f.write(f"Macro F1-Score:    {liquid_f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score: {liquid_f1_weighted:.4f}\n")
        f.write(f"Micro F1-Score:    {liquid_f1_micro:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(liquid_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(liquid_cm) + "\n")
        f.write("\n" + "="*80 + "\n")
        f.write("BOTTLE TYPE CLASSIFICATION RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Overall Accuracy:  {bottle_acc:.4f}\n")
        f.write(f"Macro F1-Score:    {bottle_f1_macro:.4f}\n")
        f.write(f"Weighted F1-Score: {bottle_f1_weighted:.4f}\n")
        f.write(f"Micro F1-Score:    {bottle_f1_micro:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(bottle_report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(bottle_cm) + "\n")
    
    print(f"\nResults saved to: {results_file}")
    
    return {
        "liquid_acc": liquid_acc,
        "liquid_f1_macro": liquid_f1_macro,
        "liquid_f1_weighted": liquid_f1_weighted,
        "liquid_f1_micro": liquid_f1_micro,
        "liquid_report": liquid_report,
        "liquid_cm": liquid_cm,
        "bottle_acc": bottle_acc,
        "bottle_f1_macro": bottle_f1_macro,
        "bottle_f1_weighted": bottle_f1_weighted,
        "bottle_f1_micro": bottle_f1_micro,
        "bottle_report": bottle_report,
        "bottle_cm": bottle_cm,
    }


# ================== Main ==================

def main():
    ap = argparse.ArgumentParser(description="Multi-task learning: liquid + bottle classification")
    
    # Data arguments
    ap.add_argument("--data_root", type=str, default="datasets_A", help="Root directory of data")
    ap.add_argument("--out_dir", type=str, default="output_mtl", help="Output directory")
    ap.add_argument("--class_prefix", type=str, default="liq-", help="Liquid class folder prefix")
    ap.add_argument("--include", type=str, default="", help="Comma-separated liquid classes to include")
    ap.add_argument("--exclude", type=str, default="", help="Comma-separated liquid classes to exclude")
    
    # Data processing
    ap.add_argument("--seq_len", type=int, default=384, help="Sequence length")
    ap.add_argument("--phase", type=str, default="all", 
                    help="Motion phase filter: 'all', 'precontact', 'closing', 'hold'")
    ap.add_argument("--features", type=str, default="ax,ay,az,gx,gy,gz", 
                    help="Comma-separated feature columns to use")
    ap.add_argument("--allow_missing_cols", action="store_true", help="Allow CSV files with missing features")
    ap.add_argument("--min_files_per_class", type=int, default=1, help="Min files per class")
    ap.add_argument("--split_mode", type=str, default="stratified_by_bid",
                    choices=["random", "stratified_by_bid", "by_device"],
                    help="Data splitting strategy")
    ap.add_argument("--train_devices", type=str, default="d1,d2,d3", help="Train devices for split_mode=by_device")
    ap.add_argument("--val_devices", type=str, default="d4", help="Val devices for split_mode=by_device")
    ap.add_argument("--train_ratio", type=float, default=0.7, help="Train/all ratio")
    ap.add_argument("--val_ratio", type=float, default=0.15, help="Val/all ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model architecture
    ap.add_argument("--d_model", type=int, default=512, help="Transformer hidden dim")
    ap.add_argument("--nhead", type=int, default=8, help="Num attention heads")
    ap.add_argument("--num_layers", type=int, default=8, help="Num transformer layers")
    ap.add_argument("--dim_ff", type=int, default=1024, help="FFN hidden dim")
    ap.add_argument("--dropout", type=float, default=0.12, help="Dropout rate")
    ap.add_argument("--pooling_strategy", type=str, default="stage_wise",
                    choices=["mean", "attention", "stage_wise", "multi"],
                    help="Pooling strategy")
    ap.add_argument("--use_residual", action="store_true", default=True, help="Use residual connections")
    ap.add_argument("--no_use_residual", dest="use_residual", action="store_false")
    
    # Training
    ap.add_argument("--epochs", type=int, default=500, help="Max epochs")
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size")
    ap.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    ap.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="Label smoothing")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping norm")
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP")
    
    # Multi-task loss weights
    ap.add_argument("--liquid_weight", type=float, default=1.0, help="Weight for liquid loss")
    ap.add_argument("--bottle_weight", type=float, default=1.0, help="Weight for bottle loss")
    
    # Early stopping
    ap.add_argument("--min_epochs", type=int, default=20, help="Min epochs before early stop")
    ap.add_argument("--patience", type=int, default=50, help="Early stop patience")
    ap.add_argument("--min_delta", type=float, default=1e-4, help="Min improvement for early stop")
    ap.add_argument("--no_early_stop", action="store_true", help="Disable early stopping")
    
    # Hardware
    ap.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    ap.add_argument("--require_cuda", action="store_true", help="Exit if CUDA not available")
    
    args = ap.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    # ===== Load and Prepare Data =====
    print(f"[INFO] Data root: {args.data_root}")
    
    all_classes = list_class_folders(args.data_root, prefix=args.class_prefix)
    classes = select_classes(all_classes, parse_csv_list(args.include), parse_csv_list(args.exclude))
    
    if len(classes) < 2:
        raise RuntimeError(f"Too few classes: {classes}")
    
    classes.sort()
    class_map = {name: i for i, name in enumerate(classes)}
    idx_to_name = {i: name for name, i in class_map.items()}
    num_liquid_classes = len(classes)
    
    print(f"[INFO] Liquid classes ({num_liquid_classes}): {classes}")
    
    # Parse phases and features
    phase_map = {
        "all": {"precontact", "closing", "hold"},
        "precontact": {"precontact"},
        "closing": {"closing"},
        "hold": {"hold"}
    }
    phase_keep = phase_map.get(args.phase, {"precontact", "closing", "hold"})
    feature_cols = parse_csv_list(args.features)
    
    # Collect CSV files
    class_to_files = {class_map[c]: [] for c in classes}
    for cname in classes:
        folder = os.path.join(args.data_root, cname)
        for fp in list_csvs(folder):
            ok, _ = validate_csv(fp, phase_keep, feature_cols, args.allow_missing_cols)
            if ok:
                class_to_files[class_map[cname]].append(fp)
    
    for i, name in enumerate(idx_to_name.values()):
        n = len(class_to_files[class_map[name]])
        print(f"[INFO] {name}: {n} files")
        if n < args.min_files_per_class:
            raise RuntimeError(f"Class {name} has too few files: {n}")
    
    # Split data
    if args.split_mode == "stratified_by_bid":
        train_items, val_items = make_splits_stratified_by_bid(
            class_to_files, args.seed, args.train_ratio, args.val_ratio
        )
    elif args.split_mode == "by_device":
        train_items, val_items = make_splits_by_device_id(
            class_to_files, args.seed,
            parse_csv_list(args.train_devices), parse_csv_list(args.val_devices)
        )
    else:
        train_items, val_items = make_splits_random_by_class(
            class_to_files, args.seed, args.train_ratio, args.val_ratio
        )
    
    print(f"[INFO] Train samples: {len(train_items)}, Val samples: {len(val_items)}")
    
    # Get bottle info from filenames
    all_bids = set()
    for path, _ in train_items + val_items:
        bid = extract_bid_from_path(path)
        if bid:
            all_bids.add(bid)
    if not all_bids:
        raise RuntimeError("No bottle type information found in file paths")
    
    bid_map = {bid: i for i, bid in enumerate(sorted(list(all_bids)))}
    idx_to_bid = {i: bid for bid, i in bid_map.items()}
    num_bottle_classes = len(bid_map)
    
    print(f"[INFO] Bottle types ({num_bottle_classes}): {bid_map}")
    
    # Compute scaler
    train_samples = []
    for path, _ in train_items:
        try:
            x = load_one_csv_as_sample(
                path, args.seq_len, phase_keep, feature_cols, args.allow_missing_cols
            )
            train_samples.append(x)
        except Exception:
            pass
    
    scaler = compute_train_scaler(train_samples)
    print(f"[INFO] Scaler computed from {len(train_samples)} samples")
    
    # ===== Training =====
    best_path, history = train_mtl_model(
        args, train_items, val_items, class_map, bid_map, phase_keep, feature_cols,
        scaler, idx_to_name, idx_to_bid, num_liquid_classes, num_bottle_classes
    )
    
    # ===== Testing =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_metrics = test_mtl_model(
        best_path, train_items, val_items, device, class_map, bid_map, args,
        idx_to_name, idx_to_bid, num_liquid_classes, num_bottle_classes,
        phase_keep, feature_cols, scaler
    )
    
    print("[INFO] Multi-task training complete!")


if __name__ == "__main__":
    main()

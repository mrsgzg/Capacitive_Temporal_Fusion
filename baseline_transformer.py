# -*- coding: utf-8 -*-
"""
Pure Transformer Baseline for Liquid Classification
Standard Transformer encoder with global average pooling
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from dataloader_mtl import (
    parse_csv_list, list_class_folders, select_classes, extract_bid_from_path,
    validate_csv, resample_to_length, load_one_csv_as_sample,
    compute_train_scaler, make_splits_stratified_by_bid,
    make_splits_random_by_class, make_splits_by_device_id,
)


class TransformerBaseline(nn.Module):
    """Pure Transformer encoder for liquid classification"""
    
    def __init__(self, seq_len, in_ch, d_model, nhead, num_layers, dim_ff, dropout, num_classes):
        super().__init__()
        self.seq_len = seq_len
        self.in_ch = in_ch
        self.d_model = d_model
        
        # Project input to d_model
        self.embed = nn.Linear(in_ch, d_model)
        
        # Positional encoding
        self.pos_enc = nn.Parameter(torch.randn(1, seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L)
        Returns:
            logits: (B, num_classes)
        """
        # (B, C, L) -> (B, L, C)
        x = x.transpose(1, 2)
        
        # Embed: (B, L, C) -> (B, L, d_model)
        x = self.embed(x)
        
        # Add positional encoding
        x = x + self.pos_enc[:, :x.size(1), :]
        
        # Transformer: (B, L, d_model) -> (B, L, d_model)
        x = self.transformer(x)
        
        # Global average pooling: (B, L, d_model) -> (B, d_model)
        x = x.mean(dim=1)
        
        # Classification: (B, d_model) -> (B, num_classes)
        return self.head(x)


class LiquidDataset(Dataset):
    """Dataset for liquid classification"""
    
    def __init__(self, items, seq_len, scaler, feature_cols, allow_missing_cols, phase_keep):
        self.items = items
        self.seq_len = seq_len
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.allow_missing_cols = allow_missing_cols
        self.phase_keep = phase_keep
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        path, label = self.items[idx]
        
        try:
            df = pd.read_csv(path)
            
            # Select columns
            missing_cols = set(self.feature_cols) - set(df.columns)
            if missing_cols and not self.allow_missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            cols_to_use = [c for c in self.feature_cols if c in df.columns]
            data = df[cols_to_use].values.astype(np.float32)
            
            # Pad if needed
            if data.shape[1] < len(self.feature_cols):
                padded = np.zeros((data.shape[0], len(self.feature_cols)), dtype=np.float32)
                padded[:, :data.shape[1]] = data
                data = padded
            
            # Select phases
            if self.phase_keep:
                start = int(len(data) * 0.1)
                end = int(len(data) * 0.9)
                data = data[start:end]
            
            # Normalize
            if self.scaler is not None:
                data = self.scaler.transform(data)
            
            # Resample to fixed length
            data = resample_to_length(data, self.seq_len)
            
            # (T, C) -> (C, T)
            x = torch.from_numpy(data.T).float()
            
            return x, label
            
        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            return torch.zeros(len(self.feature_cols), self.seq_len), label


def set_seed(seed: int = 42):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_dataloaders(train_items, val_items, seq_len, batch_size, num_workers,
                       scaler, feature_cols, allow_missing_cols, phase_keep):
    """Create dataloaders"""
    
    train_ds = LiquidDataset(train_items, seq_len, scaler, feature_cols, 
                             allow_missing_cols, phase_keep)
    val_ds = LiquidDataset(val_items, seq_len, scaler, feature_cols,
                          allow_missing_cols, phase_keep)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader


@torch.no_grad()
def evaluate_transformer(model, loader, device):
    """Evaluate Transformer model"""
    model.eval()
    preds_list, labels_list = [], []
    
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        preds_list.append(preds)
        labels_list.append(y.numpy())
    
    preds = np.concatenate(preds_list)
    labels = np.concatenate(labels_list)
    
    acc = accuracy_score(labels, preds)
    f1_mac = f1_score(labels, preds, average="macro", zero_division=0)
    
    return acc, f1_mac, preds, labels


def train_transformer(train_loader, val_loader, num_classes, device, args):
    """Train Transformer model"""
    
    # Model
    model = TransformerBaseline(
        seq_len=args.seq_len,
        in_ch=len(args.feature_cols.split(",")),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        num_classes=num_classes,
    ).to(device)
    
    print(f"[INFO] Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(train_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    criterion = nn.CrossEntropyLoss()
    grad_scaler = GradScaler()
    
    # TensorBoard
    tb_writer = SummaryWriter(log_dir=os.path.join(args.out_dir, "tensorboard"))
    
    best_acc = 0.0
    history = []
    
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_losses = []
        
        for x, y in (pbar := tqdm(train_loader, desc=f"Epoch {epoch}")):
            x = x.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            
            grad_scaler.scale(loss).backward()
            if args.grad_clip > 0:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{np.mean(train_losses):.4f}"})
        
        # Validate
        val_acc, val_f1, _, _ = evaluate_transformer(model, val_loader, device)
        
        train_loss = float(np.mean(train_losses))
        
        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(rec)
        
        tb_writer.add_scalar("loss/train", train_loss, epoch)
        tb_writer.add_scalar("metrics/val_acc", val_acc, epoch)
        tb_writer.add_scalar("lr", rec["lr"], epoch)
        
        print(f"[E{epoch:03d}] loss={train_loss:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
            }, os.path.join(args.out_dir, "best_model.pt"))
    
    tb_writer.close()
    
    return model, best_acc, history


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument("--data_root", type=str, default="datasets_New")
    parser.add_argument("--feature_cols", type=str, default="mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4")
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--phase_keep", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--allow_missing_cols", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--split_mode", type=str, default="stratified_by_bid",
                        choices=["random", "stratified_by_bid", "by_device"],
                        help="Data splitting strategy")
    parser.add_argument("--train_devices", type=str, default="d1,d2,d3",
                        help="Train devices for split_mode=by_device")
    parser.add_argument("--val_devices", type=str, default="d4",
                        help="Val devices for split_mode=by_device")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    
    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dim_ff", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.05)
    
    # Training
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--require_cuda", type=lambda x: x.lower() == "true", default=True)
    
    # Output
    parser.add_argument("--out_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Output dir
    if args.out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"baseline_transformer_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)
    
    print(f"[INFO] Output: {args.out_dir}")
    
    # Device
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")
    
    # Feature columns
    feature_cols = [c.strip() for c in args.feature_cols.split(",")]
    print(f"[INFO] Features: {feature_cols}")
    
    # Classes
    data_root = args.data_root
    all_classes = list_class_folders(data_root, prefix="liq-")
    print(f"[INFO] Classes: {len(all_classes)}")
    
    class_map = {c: i for i, c in enumerate(all_classes)}
    idx_to_name = {i: c for c, i in class_map.items()}
    
    # Data
    print("[INFO] Collecting data...")
    import glob
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
    print(f"[INFO] Train: {len(train_items)}, Val: {len(val_items)}")
    
    # Scaler
    print("[INFO] Computing scaler...")
    scaler_samples = build_scaler_samples(
        train_items, feature_cols, args.allow_missing_cols, args.phase_keep
    )
    scaler = compute_train_scaler(scaler_samples)
    
    # Dataloaders
    print("[INFO] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_items, val_items, args.seq_len, args.batch_size, args.num_workers,
        scaler, feature_cols, args.allow_missing_cols, args.phase_keep
    )
    
    # Train
    print("[INFO] Training...")
    model, best_acc, history = train_transformer(train_loader, val_loader, len(all_classes), device, args)
    
    # Final evaluation on val set
    print("[INFO] Final evaluation...")
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_model.pt"))["model"])
    val_acc, val_f1, preds, labels = evaluate_transformer(model, val_loader, device)
    
    # Reports
    report = classification_report(labels, preds, target_names=list(idx_to_name.values()),
                                   digits=4, zero_division=0)
    cm = confusion_matrix(labels, preds)
    
    print("\n" + "="*80)
    print("TRANSFORMER BASELINE - LIQUID CLASSIFICATION")
    print("="*80)
    print(f"Accuracy: {val_acc:.4f}")
    print(f"F1-Score: {val_f1:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Save results
    with open(os.path.join(args.out_dir, "results.txt"), "w") as f:
        f.write("="*80 + "\n")
        f.write("TRANSFORMER BASELINE - LIQUID CLASSIFICATION\n")
        f.write("="*80 + "\n")
        f.write(f"Accuracy: {val_acc:.4f}\n")
        f.write(f"F1-Score: {val_f1:.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    
    # Save config
    config = {
        "model": "transformer",
        "num_classes": len(all_classes),
        "classes": all_classes,
        "train_samples": len(train_items),
        "val_samples": len(val_items),
        "accuracy": float(val_acc),
        "f1_score": float(val_f1),
        "args": vars(args),
    }
    
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n[INFO] Results saved to {args.out_dir}")


if __name__ == "__main__":
    main()

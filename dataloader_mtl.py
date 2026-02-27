# -*- coding: utf-8 -*-
"""
Multi-task data loading: liquid classification + bottle type classification
"""

import os
import re
import glob
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


# ================== Utility Functions ==================

def parse_csv_list(s: str) -> List[str]:
    """Parse comma-separated string into list"""
    if s is None or str(s).strip() == "":
        return []
    parts = [p.strip() for p in str(s).split(",")]
    return [p for p in parts if p != ""]


def list_class_folders(data_root: str, prefix: str = "liq-") -> List[str]:
    """List all class folders with given prefix"""
    subs = []
    for name in os.listdir(data_root):
        p = os.path.join(data_root, name)
        if os.path.isdir(p) and name.startswith(prefix):
            subs.append(name)
    subs.sort()
    return subs


def list_csvs(folder: str) -> List[str]:
    """List all CSV files recursively"""
    files = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
    files = [f for f in files if os.path.isfile(f)]
    files.sort()
    return files


def select_classes(all_classes: List[str], include: List[str], exclude: List[str]) -> List[str]:
    """Filter classes based on include/exclude lists"""
    sel = all_classes[:]
    if include:
        inc = set(include)
        sel = [c for c in sel if c in inc]
    if exclude:
        exc = set(exclude)
        sel = [c for c in sel if c not in exc]
    return sel


def extract_bid_from_path(path: str) -> Optional[str]:
    """Extract container bid (e.g., pet01, pet02) from filename
    
    Pattern: __bid-pet01__ or __bid-glass02__
    """
    m = re.search(r"__bid-([A-Za-z0-9]+)__", path.replace("\\", "/"))
    if not m:
        return None
    return m.group(1)


def extract_device_id(path: str) -> Optional[str]:
    """Extract device ID (e.g., d1, d2, d3, d4) from filename
    
    Pattern: __d1__ or __d2__  
    """
    m = re.search(r"__d([0-9]+)__", path.replace("\\", "/"))
    if not m:
        return None
    return "d" + m.group(1)


def resample_to_length(x: np.ndarray, L: int) -> np.ndarray:
    """Resample sequence to fixed length using linear interpolation
    
    Args:
        x: (T, C) array
        L: target length
    
    Returns:
        (L, C) resampled array
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array (T,C), got {x.shape}")
    T, C = x.shape
    if T <= 0:
        raise ValueError("Empty sequence")
    if T == 1:
        return np.repeat(x, repeats=L, axis=0).astype(np.float32)
    
    old_idx = np.linspace(0, T - 1, num=T, dtype=np.float32)
    new_idx = np.linspace(0, T - 1, num=L, dtype=np.float32)
    y = np.zeros((L, C), dtype=np.float32)
    for c in range(C):
        y[:, c] = np.interp(new_idx, old_idx, x[:, c].astype(np.float32))
    return y


# ================== Data Validation & Loading ==================

def validate_csv(
    csv_path: str,
    phase_keep: Set[str],
    feature_cols: List[str],
    allow_missing_cols: bool,
    phase_col: str = "phase",
) -> Tuple[bool, str]:
    """Validate if CSV can be loaded and contains required data"""
    try:
        usecols = [phase_col] + feature_cols
        df = pd.read_csv(csv_path, usecols=usecols)
    except ValueError as e:
        if allow_missing_cols:
            try:
                df = pd.read_csv(csv_path)
            except Exception as e2:
                return False, f"read_error:{e2}"
        else:
            return False, f"read_usecols_error:{e}"
    except Exception as e:
        return False, f"read_error:{e}"

    if phase_col not in df.columns:
        return False, "missing_phase_col"

    if not allow_missing_cols:
        for c in feature_cols:
            if c not in df.columns:
                return False, f"missing_{c}"

    df = df[df[phase_col].isin(phase_keep)]
    if len(df) < 2:
        return False, f"too_few_rows_after_phase_filter:{len(df)}"

    return True, "ok"


def load_one_csv_as_sample(
    csv_path: str,
    seq_len: int,
    phase_keep: Set[str],
    feature_cols: List[str],
    allow_missing_cols: bool,
    phase_col: str = "phase",
) -> np.ndarray:
    """Load and preprocess one CSV file"""
    df = pd.read_csv(csv_path)
    
    if phase_col not in df.columns:
        raise ValueError("Missing phase column")

    if allow_missing_cols:
        for c in feature_cols:
            if c not in df.columns:
                df[c] = 0.0
    else:
        for c in feature_cols:
            if c not in df.columns:
                raise ValueError(f"Missing feature column: {c}")

    df = df[df[phase_col].isin(phase_keep)].copy()
    if len(df) < 2:
        raise ValueError(f"Too few rows after phase filter (rows={len(df)})")

    x = df[feature_cols].astype(np.float32).to_numpy()  # (T, C)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = resample_to_length(x, seq_len)  # (L, C)
    return x.astype(np.float32)


# ================== Data Scaler ==================

@dataclass
class Scaler:
    """Standardization scaler (mean, std)"""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Apply standardization"""
        return (x - self.mean[None, :]) / (self.std[None, :] + 1e-8)


def compute_train_scaler(train_samples: List[np.ndarray]) -> Scaler:
    """Compute scaler from training samples only"""
    X = np.concatenate(train_samples, axis=0)  # (N*L, C)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mean = X.mean(axis=0).astype(np.float32)
    std = X.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return Scaler(mean=mean, std=std)


# ================== MTL Dataset ==================

class LiquidBottleDataset(Dataset):
    """PyTorch Dataset for multi-task learning: liquid + bottle classification"""
    
    def __init__(
        self,
        items: List[Tuple[str, int, int]],  # (path, liquid_label, bottle_label)
        seq_len: int,
        scaler: Scaler,
        phase_keep: Set[str],
        feature_cols: List[str],
        allow_missing_cols: bool,
        augment: bool,
        strict_runtime: bool = False,
    ):
        self.items = items
        self.seq_len = seq_len
        self.scaler = scaler
        self.phase_keep = phase_keep
        self.feature_cols = feature_cols
        self.allow_missing_cols = allow_missing_cols
        self.augment = augment
        self.strict_runtime = strict_runtime
        self.n_features = len(feature_cols)

    def __len__(self):
        return len(self.items)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Data augmentation: noise + masking"""
        # Add Gaussian noise
        if random.random() < 0.8:
            x = x + np.random.normal(0.0, 0.02, size=x.shape).astype(np.float32)
        
        # Random masking
        if random.random() < 0.5:
            L = x.shape[0]
            w = random.randint(max(1, L // 40), max(2, L // 12))
            s = random.randint(0, max(0, L - w))
            x[s:s+w, :] = 0.0
        
        return x

    def __getitem__(self, idx):
        path, y_liquid, y_bottle = self.items[idx]
        try:
            x = load_one_csv_as_sample(
                path,
                seq_len=self.seq_len,
                phase_keep=self.phase_keep,
                feature_cols=self.feature_cols,
                allow_missing_cols=self.allow_missing_cols,
            )
        except Exception as e:
            if self.strict_runtime:
                raise RuntimeError(f"Failed to load sample: {path} -> {e}")
            x = np.zeros((self.seq_len, self.n_features), dtype=np.float32)

        # Standardization
        x = self.scaler.transform(x).astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        if self.augment:
            x = self._augment(x)

        # (L, C) -> (C, L) for Conv1d
        x_t = torch.from_numpy(x).transpose(0, 1).contiguous()
        return x_t, torch.tensor(y_liquid, dtype=torch.long), torch.tensor(y_bottle, dtype=torch.long), path


# ================== Data Splitting ==================

def split_list(
    files: List[str],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[str], List[str], List[str]]:
    """Split list into train/val/test"""
    files = files[:]
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    train = files[:n_train]
    val = files[n_train:n_train + n_val]
    test = files[n_train + n_val:]
    return train, val, test


def make_splits_random_by_class(
    class_to_files: Dict[int, List[str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split randomly inside each class"""
    rng = random.Random(seed)
    train, val = [], []

    for y, files in class_to_files.items():
        tr, va, _ = split_list(files, rng, train_ratio, val_ratio)
        train += [(f, y) for f in tr]
        val += [(f, y) for f in va]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_splits_stratified_by_bid(
    class_to_files: Dict[int, List[str]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Split inside each (class, bid) group to keep bid distribution stable"""
    rng = random.Random(seed)
    train, val = [], []

    for y, files in class_to_files.items():
        groups: Dict[str, List[str]] = {}
        for f in files:
            bid = extract_bid_from_path(f) or "UNKNOWN"
            groups.setdefault(bid, []).append(f)

        for _, gfiles in groups.items():
            tr, va, _ = split_list(gfiles, rng, train_ratio, val_ratio)
            train += [(f, y) for f in tr]
            val += [(f, y) for f in va]

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_splits_by_device_id(
    class_to_files: Dict[int, List[str]],
    seed: int,
    train_devices: List[str],
    val_devices: List[str],
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Split by device ID: e.g., d1,d2,d3 as train; d4 as validation
    Pattern: __d1__, __d2__, __d3__, __d4__
    """
    train_devs = set(train_devices)
    val_devs = set(val_devices)
    train, val = [], []

    for y, files in class_to_files.items():
        for f in files:
            dev_id = extract_device_id(f)
            if dev_id in train_devs:
                train.append((f, y))
            elif dev_id in val_devs:
                val.append((f, y))

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(
            f"Device-based split resulted in empty split. "
            f"Train devices: {train_devices}, Val devices: {val_devices}"
        )
    
    return train, val


# ================== DataLoader Creation ==================

def create_dataloaders_mtl(
    train_items: List[Tuple[str, int]],
    val_items: List[Tuple[str, int]],
    seq_len: int,
    batch_size: int,
    num_workers: int,
    scaler: Scaler,
    phase_keep: Set[str],
    feature_cols: List[str],
    allow_missing_cols: bool,
) -> Tuple[DataLoader, DataLoader, Dict[str, int], Dict[int, str]]:
    """Create train and validation dataloaders with bottle type labels
    
    Returns:
        Tuple of (loader_train, loader_val, bid_map, idx_to_bid)
        - bid_map: {"pet01": 0, "pet02": 1, "glass01": 2, "glass02": 3}
        - idx_to_bid: {0: "pet01", 1: "pet02", ...}
    """
    
    # Build bid mapping from all items
    all_bids = set()
    for path, _ in train_items + val_items:
        bid = extract_bid_from_path(path)
        if bid:
            all_bids.add(bid)
    
    all_bids = sorted(list(all_bids))
    bid_map = {bid: i for i, bid in enumerate(all_bids)}
    idx_to_bid = {i: bid for bid, i in bid_map.items()}
    
    # Augment items with bottle labels
    def add_bottle_labels(items):
        augmented = []
        for path, y_liquid in items:
            bid = extract_bid_from_path(path)
            y_bottle = bid_map.get(bid, -1)
            if y_bottle >= 0:  # Only add items with valid bottle type
                augmented.append((path, y_liquid, y_bottle))
        return augmented
    
    train_items_mtl = add_bottle_labels(train_items)
    val_items_mtl = add_bottle_labels(val_items)
    
    print(f"[INFO] Bottle type mapping: {bid_map}")
    print(f"[INFO] Training samples with bottles: {len(train_items_mtl)}/{len(train_items)}")
    print(f"[INFO] Validation samples with bottles: {len(val_items_mtl)}/{len(val_items)}")
    
    ds_train = LiquidBottleDataset(
        train_items_mtl, seq_len, scaler, phase_keep, feature_cols,
        allow_missing_cols, augment=True, strict_runtime=False
    )
    ds_val = LiquidBottleDataset(
        val_items_mtl, seq_len, scaler, phase_keep, feature_cols,
        allow_missing_cols, augment=False, strict_runtime=False
    )

    loader_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(num_workers > 0)
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return loader_train, loader_val, bid_map, idx_to_bid

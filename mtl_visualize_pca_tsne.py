# -*- coding: utf-8 -*-
"""
PCA/t-SNE visualization for MTL model embeddings on validation set.
"""

import os
import json
import csv
import time
import random
import argparse
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataloader_mtl import (
    parse_csv_list, list_class_folders, list_csvs, select_classes,
    extract_bid_from_path, validate_csv, load_one_csv_as_sample,
    compute_train_scaler, create_dataloaders_mtl,
    make_splits_random_by_class, make_splits_stratified_by_bid, make_splits_by_device_id,
)
from CNN_transformer_mtl import CNNTransformerMTL


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_keep_set(phase: str):
    phase_map = {
        "all": {"precontact", "closing", "hold"},
        "precontact": {"precontact"},
        "closing": {"closing"},
        "hold": {"hold"},
    }
    return phase_map.get(phase, {"precontact", "closing", "hold"})


def extract_shared_features(model: CNNTransformerMTL, x: torch.Tensor) -> torch.Tensor:
    x = model.initial_conv(x)
    x = model.conv_scale1(x)
    if model.use_residual:
        x = model.residual_block(x)
    x = model.conv_scale2(x)
    x = x.transpose(1, 2)
    x = model.positional_encoding(x)
    x = model.transformer_encoder(x)
    if model.pooling_strategy == "mean":
        x = x.transpose(1, 2)
        x = model.pooling(x).squeeze(-1)
    else:
        x = model.pooling(x)
    return x


def plot_scatter(emb, labels, label_names, title, out_path, show_legend):
    num_classes = len(label_names)
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        idx = labels == i
        if not np.any(idx):
            continue
        plt.scatter(
            emb[idx, 0], emb[idx, 1],
            s=8, alpha=0.7,
            color=cmap(i % 20),
            label=label_names[i]
        )
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    if show_legend:
        plt.legend(markerscale=2, fontsize=8, loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Robot_hand/MLC/output_mtl_20260227_073453/best_mtl.pt")
    parser.add_argument("--data_root", type=str, default="/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Robot_hand/datasets_New")
    parser.add_argument("--class_prefix", type=str, default="liq-")
    parser.add_argument("--include", type=str, default="")
    parser.add_argument("--exclude", type=str, default="")
    parser.add_argument("--features", type=str, default="mcap_delta_1,mcap_delta_2,mcap_delta_3,mcap_delta_4")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "precontact", "closing", "hold"])
    parser.add_argument("--allow_missing_cols", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--split_mode", type=str, default="stratified_by_bid",
                        choices=["random", "stratified_by_bid", "by_device"])
    parser.add_argument("--train_devices", type=str, default="d1,d2,d3")
    parser.add_argument("--val_devices", type=str, default="d4")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.3)
    parser.add_argument("--seq_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=256)
    parser.add_argument("--tsne_perplexity", type=int, default=30)
    parser.add_argument("--tsne_iter", type=int, default=1000)
    parser.add_argument("--tsne_lr", type=float, default=200.0)
    parser.add_argument("--out_dir", type=str, default='/mnt/iusers01/fatpou01/compsci01/k09562zs/scratch/Robot_hand/MLC/visualize_mtl_20260227_073453')
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--legend_liquid", action="store_true")

    args = parser.parse_args()
    set_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    if args.out_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.out_dir = f"mtl_vis_{timestamp}"
    os.makedirs(args.out_dir, exist_ok=True)

    feature_cols = parse_csv_list(args.features)
    phase_keep = phase_keep_set(args.phase)

    all_classes = list_class_folders(args.data_root, prefix=args.class_prefix)
    classes = select_classes(all_classes, parse_csv_list(args.include), parse_csv_list(args.exclude))
    classes.sort()
    class_map = {name: i for i, name in enumerate(classes)}
    idx_to_name = {i: name for name, i in class_map.items()}

    class_to_files = {class_map[c]: [] for c in classes}
    for cname in classes:
        folder = os.path.join(args.data_root, cname)
        for fp in list_csvs(folder):
            ok, _ = validate_csv(fp, phase_keep, feature_cols, args.allow_missing_cols)
            if ok:
                class_to_files[class_map[cname]].append(fp)

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

    # Compute scaler from training data
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

    _, loader_val, bid_map, idx_to_bid = create_dataloaders_mtl(
        train_items, val_items, args.seq_len, args.batch_size, args.num_workers,
        scaler, phase_keep, feature_cols, args.allow_missing_cols
    )

    num_liquid_classes = len(classes)
    num_bottle_classes = len(bid_map)

    # Build model from checkpoint args
    model = CNNTransformerMTL(
        seq_len=ckpt_args.get("seq_len", args.seq_len),
        in_ch=len(feature_cols),
        d_model=ckpt_args.get("d_model", 512),
        nhead=ckpt_args.get("nhead", 8),
        num_layers=ckpt_args.get("num_layers", 8),
        dim_ff=ckpt_args.get("dim_ff", 1024),
        dropout=ckpt_args.get("dropout", 0.12),
        num_liquid_classes=num_liquid_classes,
        num_bottle_classes=num_bottle_classes,
        pooling_strategy=ckpt_args.get("pooling_strategy", "stage_wise"),
        use_residual=ckpt_args.get("use_residual", True),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    model.to(device)

    feats = []
    y_liquid = []
    y_bottle = []
    paths = []

    with torch.no_grad():
        for x, yl, yb, p in loader_val:
            x = x.to(device, non_blocking=True)
            f = extract_shared_features(model, x).cpu().numpy()
            feats.append(f)
            y_liquid.append(yl.numpy())
            y_bottle.append(yb.numpy())
            paths.extend(list(p))

    feats = np.concatenate(feats, axis=0)
    y_liquid = np.concatenate(y_liquid, axis=0)
    y_bottle = np.concatenate(y_bottle, axis=0)

    # Subsample for visualization
    if args.max_samples > 0 and feats.shape[0] > args.max_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(feats.shape[0], size=args.max_samples, replace=False)
        feats = feats[idx]
        y_liquid = y_liquid[idx]
        y_bottle = y_bottle[idx]
        paths = [paths[i] for i in idx]

    # PCA
    pca = PCA(n_components=2, random_state=args.seed)
    emb_pca = pca.fit_transform(feats)

    # t-SNE
    perplexity = min(args.tsne_perplexity, max(5, (len(feats) - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=args.tsne_iter,
        learning_rate=args.tsne_lr,
        init="pca",
        random_state=args.seed,
    )
    emb_tsne = tsne.fit_transform(feats)

    # Plots
    plot_scatter(
        emb_pca, y_liquid, [idx_to_name[i] for i in range(num_liquid_classes)],
        "PCA - Liquid", os.path.join(args.out_dir, "pca_liquid.png"), args.legend_liquid
    )
    plot_scatter(
        emb_pca, y_bottle, [idx_to_bid[i] for i in range(num_bottle_classes)],
        "PCA - Bottle", os.path.join(args.out_dir, "pca_bottle.png"), True
    )
    plot_scatter(
        emb_tsne, y_liquid, [idx_to_name[i] for i in range(num_liquid_classes)],
        "TSNE - Liquid", os.path.join(args.out_dir, "tsne_liquid.png"), args.legend_liquid
    )
    plot_scatter(
        emb_tsne, y_bottle, [idx_to_bid[i] for i in range(num_bottle_classes)],
        "TSNE - Bottle", os.path.join(args.out_dir, "tsne_bottle.png"), True
    )

    # Save embeddings
    csv_path = os.path.join(args.out_dir, "embeddings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "method", "x", "y", "liquid_label", "liquid_name",
            "bottle_label", "bottle_name", "path"
        ])
        for i in range(len(feats)):
            writer.writerow([
                "pca", emb_pca[i, 0], emb_pca[i, 1],
                int(y_liquid[i]), idx_to_name[int(y_liquid[i])],
                int(y_bottle[i]), idx_to_bid[int(y_bottle[i])],
                paths[i],
            ])
        for i in range(len(feats)):
            writer.writerow([
                "tsne", emb_tsne[i, 0], emb_tsne[i, 1],
                int(y_liquid[i]), idx_to_name[int(y_liquid[i])],
                int(y_bottle[i]), idx_to_bid[int(y_bottle[i])],
                paths[i],
            ])

    # Save summary
    summary = {
        "ckpt": args.ckpt,
        "num_samples": int(len(feats)),
        "num_liquid_classes": num_liquid_classes,
        "num_bottle_classes": num_bottle_classes,
        "phase": args.phase,
        "features": feature_cols,
        "split_mode": args.split_mode,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "tsne_perplexity": perplexity,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved plots and embeddings to: {args.out_dir}")


if __name__ == "__main__":
    main()

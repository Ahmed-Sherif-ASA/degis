# main.py
"""
Run exactly the same training you had in the notebook.

Usage (examples):
    python main.py                           # uses HCL 514 by default
    python main.py --hist-kind lab514
    python main.py --hist-kind rgb512 --epochs 120 --batch-size 192

Checkpoints:
    best_color_head.pth
    best_rest_head.pth
"""

import os
import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import json, time, platform
from utils.logger import MetricsLogger, short_git_hash

import config
from data.dataset import PrecompClipColorDataset
from models.color_heads import ColorHead, RestHead
from models.train_color import train_color_disentanglement

torch.backends.cudnn.benchmark = True
torch.manual_seed(42); np.random.seed(42)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hist-kind", choices=["rgb512","lab514","hcl514"], default="hcl514")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--blur", type=float, default=0.05)
    p.add_argument("--lambda-ortho", type=float, default=0.1)
    p.add_argument("--top-k", type=int, default=None)  # None ⇒ full target
    p.add_argument("--weighting", action="store_true") # rarity weighting off by default
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --- make run directory
    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{args.hist_kind}_tk{args.top_k or 'all'}_b{args.batch_size}"
    outdir = os.path.join("runs", f"{run_name}-{stamp}")
    os.makedirs(outdir, exist_ok=True)
    print("Run dir:", outdir)

    logger = MetricsLogger(outdir=outdir)

    # paths
    emb_path = getattr(config, "EMBEDDINGS_TARGET_PATH",
               getattr(config, "EMBEDDINGS_PATH", None))
    assert emb_path, "Set EMBEDDINGS_TARGET_PATH in config.py"

    hist_map = {
        "rgb512": config.COLOR_HIST_PATH_RGB,
        "lab514": config.COLOR_HIST_PATH_LAB_514,
        "hcl514": config.COLOR_HIST_PATH_HCL_514,
    }
    hist_path = hist_map[args.hist_kind]

    # load into RAM (float32, no copy if possible)
    emb  = np.load(emb_path).astype(np.float32, copy=False)
    hist = np.load(hist_path).astype(np.float32, copy=False)
    assert emb.shape[0] == hist.shape[0], "Embeddings and histograms must share N"

    N, clip_dim = emb.shape
    hist_dim    = hist.shape[1]
    print(f"Loaded → emb: {emb.shape} | hist: {hist.shape} | kind={args.hist_kind}")

    # split
    idx_train, idx_val = train_test_split(np.arange(N), test_size=0.2, random_state=42)

    # datasets / loaders
    train_ds = PrecompClipColorDataset(idx_train, emb, hist)
    val_ds   = PrecompClipColorDataset(idx_val,   emb, hist)

    # use the CLI sizes you already parse
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=16, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.val_batch_size, shuffle=False,
                            num_workers=16, pin_memory=True)

    # optional rarity weighting
    weights_vec = None
    if args.weighting:
        counts = hist.sum(0).astype(np.float64) + 1e-6
        alpha  = 0.5
        w = (1.0 / (counts ** alpha))
        w = (w / w.mean()).astype(np.float32)
        weights_vec = torch.tensor(w, device=device)

    # models
    color_head = ColorHead(clip_dim=clip_dim, hist_dim=hist_dim).to(device)
    rest_head  = RestHead(clip_dim=clip_dim).to(device)

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    logger.set_meta(
        run_name=run_name,
        hist_kind=args.hist_kind,
        top_k=args.top_k, blur=args.blur,
        lambda_ortho=args.lambda_ortho, lambda_leak=0.25,   # your current default
        epochs=args.epochs, batch_size=args.batch_size, val_batch_size=args.val_batch_size,
        optimizer="AdamW", lr=args.lr, weight_decay=args.wd,
        device=str(device), gpu_name=gpu_name, seed=42,
        param_count_color=MetricsLogger.param_count(color_head),
        param_count_rest=MetricsLogger.param_count(rest_head),
        git_commit=short_git_hash(),
    )

    # train (same defaults as notebook)
    train_color_disentanglement(
        train_loader=train_loader,
        val_loader=val_loader,
        color_head=color_head,
        rest_head=rest_head,
        device=device,
        num_epochs=args.epochs,
        lambda_ortho=args.lambda_ortho,
        top_k=args.top_k,
        use_weighting=bool(args.weighting),
        weights_vec=weights_vec,
        T_min=1.0,
        blur=args.blur,
        lr=args.lr, wd=args.wd,
        save_prefix = os.path.join(outdir, "best_color_head_tmp"),
        logger=logger,                   # NEW
    )

if __name__ == "__main__":
    main()

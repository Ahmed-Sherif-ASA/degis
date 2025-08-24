# edge_train_main.py
"""
Train an edge/layout head from precomputed CLIP embeddings and edge maps.

- Uses your config paths:
    EMBEDDINGS_TARGET_PATH
    EDGE_MAPS_PATH
- Edge maps may be uint8 (0–255) or float; dataset converts to float32 in [0,1].
- Defaults to memory-mapped loads; add --in-ram to copy into RAM first.
- Saves best weights to best_edge_head.pth

Example:
    poetry run python edge_train_main.py \
        --epochs 200 --batch 512 --val-batch 1024 --num-workers 8
"""
import argparse, numpy as np, torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import config
from data.dataset import PrecompClipEdgeDataset
from models.edge_heads import EdgeHead, RestHead
from models.train_edge import train_edge_decoder

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--val-batch", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--lambda-ortho", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--in-ram", action="store_true")  # copy arrays fully
    p.add_argument("--ckpt", default="best_edge_head.pth")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    emb_path = getattr(config, "EMBEDDINGS_TARGET_PATH",
               getattr(config, "EMBEDDINGS_PATH", None))
    edge_path = config.EDGE_MAPS_PATH

    # load arrays
    emb  = np.load(emb_path,  mmap_mode=None if args.in_ram else "r")
    edges = np.load(edge_path, mmap_mode=None if args.in_ram else "r")

    # optional RAM copy + dtype normalization for edges if desired
    if args.in_ram:
        if emb.dtype != np.float32:
            emb = emb.astype(np.float32, copy=True)
        if edges.dtype != np.float32:
            edges = edges.astype(np.float32, copy=True)

    N, clip_dim = emb.shape
    edge_dim = edges.shape[1]
    print(f"Loaded → emb: {emb.shape} | edges: {edges.shape}")

    idx_train, idx_val = train_test_split(np.arange(N), test_size=0.2, random_state=42)

    train_ds = PrecompClipEdgeDataset(idx_train, emb, edges, normalize_edges=True)
    val_ds   = PrecompClipEdgeDataset(idx_val,   emb, edges, normalize_edges=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        pin_memory_device="cuda", persistent_workers=True, prefetch_factor=6
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.val_batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        pin_memory_device="cuda", persistent_workers=True, prefetch_factor=6
    )


    edge_head = EdgeHead(clip_dim=clip_dim, edge_dim=edge_dim).to(device)
    rest_head = RestHead(clip_dim=clip_dim).to(device)

    train_edge_decoder(
        train_loader=train_loader,
        val_loader=val_loader,
        edge_head=edge_head,
        rest_head=rest_head,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        lambda_ortho=args.lambda_ortho,
        patience=args.patience,
        ckpt_path=args.ckpt,
    )

if __name__ == "__main__":
    main()

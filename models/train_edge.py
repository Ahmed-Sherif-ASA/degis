# models/train_edge.py
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

def orthogonality_loss(h1, h2):
    # minimize cross-covariance (rough disentanglement)
    h1c = h1 - h1.mean(dim=0)
    h2c = h2 - h2.mean(dim=0)
    denom = max(1, (h1.size(0) - 1))
    cov = (h1c.T @ h2c) / denom
    return torch.norm(cov, p="fro") ** 2

@torch.no_grad()
def eval_edge(val_loader, edge_head, device):
    edge_head.eval()
    total = 0.0
    for z, e in val_loader:
        z = z.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        _, pred, _ = edge_head(z)
        total += F.mse_loss(pred, e).item()
    return total / max(1, len(val_loader))

def train_edge_decoder(
    train_loader,
    val_loader,
    edge_head,
    rest_head,
    *,
    device,
    num_epochs=200,
    lr=1e-4,
    weight_decay=0.0,
    lambda_ortho=0.1,
    patience=10,
    ckpt_path="best_edge_head.pth",
):
    opt = torch.optim.AdamW(
        list(edge_head.parameters()) + list(rest_head.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    best_val = float("inf")
    bad = 0

    for ep in range(1, num_epochs + 1):
        edge_head.train(); rest_head.train()
        running = 0.0
        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {ep:02d}/{num_epochs} [TRAIN]", dynamic_ncols=True)
        for z, e in pbar:
            z = z.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)

            h_edge, pred, _ = edge_head(z)
            h_rest = rest_head(z)

            loss_edge  = F.mse_loss(pred, e)
            loss_ortho = orthogonality_loss(h_edge, h_rest)
            loss = loss_edge + lambda_ortho * loss_ortho

            opt.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(edge_head.parameters(), 1.0)
            opt.step()

            running += float(loss_edge.item())
            pbar.set_postfix({"MSE": f"{running/max(1,pbar.n):.4f}"})

        train_mse = running / max(1, len(train_loader))
        val_mse = eval_edge(val_loader, edge_head, device)
        print(f"Epoch {ep:02d}  train MSE={train_mse:.4f}  val MSE={val_mse:.4f}")

        if val_mse < best_val - 1e-6:
            best_val, bad = val_mse, 0
            torch.save(edge_head.state_dict(), ckpt_path)
            print("✓ saved best")
        else:
            bad += 1
            print(f"  no improvement ({bad}/{patience})")
            if bad >= patience:
                print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
                break

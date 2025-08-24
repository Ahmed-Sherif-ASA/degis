# models/train.py
import torch
# torch.set_float32_matmul_precision("high")  # enables TF32 on Ampere/Ada
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

try:
    import geomloss
except Exception as e:
    raise RuntimeError("Please `pip install geomloss`") from e


def orthogonality_loss(h1, h2):
    # minimize cross-covariance between two heads (rough disentanglement)
    h1_c = h1 - h1.mean(dim=0)
    h2_c = h2 - h2.mean(dim=0)
    denom = max(1, (h1.size(0) - 1))
    cov  = (h1_c.T @ h2_c) / denom
    return torch.norm(cov, p="fro") ** 2


@torch.no_grad()
def eval_model(loader, color_head, sinkhorn, device):
    color_head.eval()
    total = 0.0
    for z, hist in loader:
        z    = z.to(device, non_blocking=True)
        hist = hist.to(device, non_blocking=True)
        _, probs, _ = color_head(z)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)
        total += sinkhorn(probs, hist).item()
    return total / max(1, len(loader))


def train_color_disentanglement(
    train_loader,
    val_loader,
    color_head,
    rest_head,
    *,
    device,
    num_epochs    = 200,
    lambda_ortho  = 0.1,
    top_k         = None,      # None => full target
    use_weighting = False,
    weights_vec   = None,      # torch [D] on device when use_weighting=True
    T_min         = 1.0,       # no temperature anneal by default
    blur          = 0.05,      # Sinkhorn blur
    lr            = 1e-3,
    wd            = 1e-2,
    patience      = 10,        # early stop patience
    save_prefix   = "best_color_head",
    metric_label  = "EMD"
):
    # sinkhorn  = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur)
    SINKHORN_BACKEND = "tensorized"
    sinkhorn = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, backend=SINKHORN_BACKEND)
    params    = list(color_head.parameters()) + list(rest_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    best_val = float("inf")
    bad = 0

    for ep in range(1, num_epochs + 1):
        # linear anneal 1.0 → T_min
        T = 1.0 - (1.0 - T_min) * (ep - 1) / max(1, (num_epochs - 1))

        color_head.train(); rest_head.train()
        running_color = 0.0
        running_bce   = 0.0
        n_steps       = 0

        pbar = tqdm(
            train_loader, total=len(train_loader),
            desc=f"Epoch {ep:02d}/{num_epochs}", dynamic_ncols=True
        )

        for z, hist in pbar:
            z    = z.to(device, non_blocking=True)
            hist = hist.to(device, non_blocking=True)

            # Top-k mask (optional)
            if top_k is None:
                hist_target = hist
            else:
                _, idx_desc = hist.sort(dim=1, descending=True)
                keep_idx    = idx_desc[:, :top_k]
                mask        = torch.zeros_like(hist).scatter_(1, keep_idx, 1.0)
                hist_target = hist * mask
                hist_target = hist_target / (hist_target.sum(dim=1, keepdim=True) + 1e-8)

            logits, _, c_emb = color_head(z)
            probs = torch.softmax(logits / T, dim=1)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

            # Optional rarity weighting
            if use_weighting:
                p_w = probs * weights_vec
                h_w = hist_target * weights_vec
                p_w = p_w / (p_w.sum(dim=1, keepdim=True) + 1e-8)
                h_w = h_w / (h_w.sum(dim=1, keepdim=True) + 1e-8)
                loss_color = sinkhorn(p_w, h_w)
            else:
                loss_color = sinkhorn(probs, hist_target)

            # Aux losses
            loss_ortho = orthogonality_loss(c_emb, rest_head(z))
            loss_recon = F.mse_loss(rest_head(z), z.detach())

            # Diagnostic BCE (not in loss)
            bce_metric = F.binary_cross_entropy(probs, hist)

            loss = loss_color + lambda_ortho * loss_ortho + 0.1 * loss_recon

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(color_head.parameters(), 1.0)
            optimizer.step()

            running_color += float(loss_color.item())
            running_bce   += float(bce_metric.item())
            n_steps       += 1

            pbar.set_postfix({metric_label: f"{running_color/n_steps:.4f}", "T": f"{T:.2f}"})

        # epoch metrics
        train_color = running_color / max(1, n_steps)
        train_bce   = running_bce   / max(1, n_steps)

        val_color = eval_model(val_loader, color_head, sinkhorn, device)

        print(f"Epoch {ep:02d}  train {metric_label}={train_color:.4f}  "
              f"val {metric_label}={val_color:.4f}  (diag BCE={train_bce:.4f})")

        # checkpoint + early stop
        if val_color < best_val - 1e-6:
            best_val = val_color
            bad = 0
            torch.save(color_head.state_dict(), f"{save_prefix}.pth")
            torch.save(rest_head.state_dict(),  f"{save_prefix.replace('color','rest')}.pth")
            print("✓ saved best")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
                break

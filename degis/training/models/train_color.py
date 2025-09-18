# models/train_color.py
import os, json, time
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

try:
    import geomloss
except Exception as e:
    raise RuntimeError("Please `pip install geomloss`") from e


def orthogonality_loss(h1, h2):
    h1_c = h1 - h1.mean(dim=0)
    h2_c = h2 - h2.mean(dim=0)
    denom = max(1, (h1.size(0) - 1))
    cov  = (h1_c.T @ h2_c) / denom
    return torch.norm(cov, p="fro") ** 2


@torch.no_grad()
def eval_model(loader, color_head, sinkhorn, device, top_k=None):
    color_head.eval()
    total = 0.0
    for z, hist in loader:
        z    = z.to(device, non_blocking=True)
        hist = hist.to(device, non_blocking=True)

        logits, probs, _ = color_head(z)
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

        if top_k is None:
            hist_target = hist
            p_use = probs
        else:
            _, idx_desc = hist.sort(dim=1, descending=True)
            keep_idx    = idx_desc[:, :top_k]
            mask        = torch.zeros_like(hist).scatter_(1, keep_idx, 1.0).bool()
            hist_target = (hist * mask)
            hist_target = hist_target / (hist_target.sum(dim=1, keepdim=True) + 1e-8)
            p_use       = probs.masked_fill(~mask, 0.0)
            p_use       = p_use / (p_use.sum(dim=1, keepdim=True) + 1e-8)

        total += geomloss.SamplesLoss("sinkhorn", p=2, blur=sinkhorn.blur,
                                      backend=sinkhorn.backend)(p_use, hist_target).item()
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
    lambda_consistency = 0.1,  # New: consistency loss weight
    lambda_leak   = 0.25,
    top_k         = None,
    use_weighting = False,
    weights_vec   = None,
    T_min         = 1.0,
    blur          = 0.05,
    lr            = 1e-3,
    wd            = 1e-2,
    patience      = 10,
    save_prefix   = "best_color_head",
    metric_label  = "Sinkhorn",
    logger        = True,     # <— NEW
):
    # Sinkhorn loss (keep backend/blur on the object so eval can reuse them)
    sinkhorn = geomloss.SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    sinkhorn.blur = blur; sinkhorn.backend = "tensorized"

    params    = list(color_head.parameters()) + list(rest_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    best_val = float("inf")
    bad = 0
    outdir = os.path.dirname(save_prefix)

    for ep in range(1, num_epochs + 1):
        t_ep = time.time()
        # linear anneal 1.0 → T_min
        T = 1.0 - (1.0 - T_min) * (ep - 1) / max(1, (num_epochs - 1))

        color_head.train(); rest_head.train()
        running_color = running_bce = running_leak = running_ortho = running_recon = running_consistency = 0.0
        running_gn    = 0.0
        n_steps       = 0

        pbar = tqdm(train_loader, total=len(train_loader),
                    desc=f"Epoch {ep:02d}/{num_epochs}", dynamic_ncols=True)

        for z, hist in pbar:
            z    = z.to(device, non_blocking=True)
            hist = hist.to(device, non_blocking=True)

            # Top-k mask (optional)
            if top_k is None:
                hist_target = hist
                mask = None
            else:
                _, idx_desc = hist.sort(dim=1, descending=True)
                keep_idx    = idx_desc[:, :top_k]
                mask        = torch.zeros_like(hist).scatter_(1, keep_idx, 1.0).bool()
                hist_target = (hist * mask)
                hist_target = hist_target / (hist_target.sum(dim=1, keepdim=True) + 1e-8)

            logits, _, c_emb = color_head(z)
            probs = torch.softmax(logits / T, dim=1)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-8)

            if mask is None:
                p_use = probs
            else:
                p_use = probs.masked_fill(~mask, 0.0)
                p_use = p_use / (p_use.sum(dim=1, keepdim=True) + 1e-8)

            # Optional rarity weighting
            if use_weighting:
                p_w = p_use * weights_vec
                h_w = hist_target * weights_vec
                p_w = p_w / (p_w.sum(dim=1, keepdim=True) + 1e-8)
                h_w = h_w / (h_w.sum(dim=1, keepdim=True) + 1e-8)
                loss_color = sinkhorn(p_w, h_w)
            else:
                loss_color = sinkhorn(p_use, hist_target)

            # Leakage penalty (mass outside top-k)
            leak = torch.zeros(z.size(0), device=z.device)
            if mask is not None:
                leak = probs.masked_fill(mask, 0.0).sum(dim=1)
            loss_leak = leak.mean()

            # Aux losses
            rest = rest_head(z)
            loss_ortho = orthogonality_loss(c_emb, rest)
            loss_recon = F.mse_loss(rest, z.detach())
            
            # Consistency loss: color + rest ≈ original embedding
            reconstructed = c_emb + rest
            loss_consistency = F.mse_loss(reconstructed, z)

            # Diagnostic only
            bce_metric = F.binary_cross_entropy(probs, hist)

            # Final loss
            loss = loss_color + lambda_leak * loss_leak + lambda_ortho * loss_ortho + 0.1 * loss_recon + lambda_consistency * loss_consistency

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn = float(clip_grad_norm_(params, 1.0))
            optimizer.step()

            if logger is not None:
                logger.bump_iterations(batch_size=z.size(0))

            # running means
            running_color += float(loss_color.item())
            running_bce   += float(bce_metric.item())
            running_leak  += float(loss_leak.item())
            running_ortho += float(loss_ortho.item())
            running_recon += float(loss_recon.item())
            running_consistency += float(loss_consistency.item())
            running_gn    += gn
            n_steps       += 1

            pbar.set_postfix({metric_label: f"{running_color/n_steps:.4f}", "T": f"{T:.2f}"})

        # epoch metrics
        train_color = running_color / max(1, n_steps)
        train_bce   = running_bce   / max(1, n_steps)
        train_leak  = running_leak  / max(1, n_steps)
        train_ortho = running_ortho / max(1, n_steps)
        train_recon = running_recon / max(1, n_steps)
        train_consistency = running_consistency / max(1, n_steps)
        grad_norm   = running_gn    / max(1, n_steps)

        val_color = eval_model(val_loader, color_head, sinkhorn, device, top_k=top_k)

        print(f"Epoch {ep:02d}  train {metric_label}={train_color:.4f}  "
              f"val {metric_label}={val_color:.4f}  (diag BCE={train_bce:.4f})")

        # per-epoch CSV row
        if logger is not None:
            epoch_seconds = float(time.time() - t_ep)
            total_loss = train_color + 0.25*train_leak + 0.1*train_recon + 0.1*train_ortho + lambda_consistency*train_consistency
            logger.log_epoch(
                epoch=ep,
                train_sinkhorn=train_color, val_sinkhorn=val_color,
                loss=float(total_loss),  # Required field
                lr=float(optimizer.param_groups[0]["lr"]),
                wall_time=epoch_seconds,  # Required field
                diag_bce=train_bce,
                leak_loss=train_leak, ortho_loss=train_ortho, recon_loss=train_recon, consistency_loss=train_consistency,
                total_loss=float(total_loss),  # Keep original too
                grad_norm=grad_norm,
                it_per_sec=float(logger.it_per_sec()),
                samples_per_sec=float(logger.samples_per_sec()),
            )

        # checkpoint + early stop
        if val_color < best_val - 1e-6:
            best_val = val_color
            bad = 0
            color_pth = f"{save_prefix}.pth"
            rest_pth  = f"{save_prefix.replace('color','rest')}.pth"
            torch.save(color_head.state_dict(), color_pth)
            torch.save(rest_head.state_dict(),  rest_pth)
            
            # Save safetensors + hashes
            try:
                from safetensors.torch import save_file
                import hashlib
                
                color_st = f"{save_prefix}.safetensors"
                rest_st = f"{save_prefix.replace('color','rest')}.safetensors"
                
                save_file(color_head.state_dict(), color_st)
                save_file(rest_head.state_dict(), rest_st)
                
                def sha256(path):
                    h = hashlib.sha256()
                    with open(path, "rb") as f:
                        for chunk in iter(lambda: f.read(1<<20), b""):
                            h.update(chunk)
                    return h.hexdigest()
                
                manifest = {
                    "color_pth_sha256": sha256(color_pth),
                    "rest_pth_sha256":  sha256(rest_pth),
                    "color_st_sha256":  sha256(color_st),
                    "rest_st_sha256":   sha256(rest_st),
                }
                with open(os.path.join(outdir, "checkpoint_hashes.json"), "w") as f:
                    json.dump(manifest, f, indent=2)
                    
            except ImportError:
                print("Warning: safetensors not available, skipping .safetensors files")
            except Exception as e:
                print(f"Warning: Failed to create safetensors/hashes: {e}")
                import traceback
                traceback.print_exc()
            
            print("✓ saved best")

            # JSON summary for the paper
            if logger is not None:
                best_json = {
                    "best_epoch": ep,
                    "best_val_sinkhorn": float(val_color),
                    "train_sinkhorn_at_best": float(train_color),
                    "diag_bce_at_best": float(train_bce),
                    "checkpoint_color": os.path.basename(color_pth),
                    "checkpoint_rest":  os.path.basename(rest_pth),
                }
                best_json.update(logger.meta)
                with open(os.path.join(outdir, "best_summary.json"), "w") as f:
                    json.dump(best_json, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {ep} (best val={best_val:.4f})")
                break

    # final run metadata
    if logger is not None:
        meta = logger.meta.copy()
        meta.update({
            "wall_seconds": logger.wall_seconds(),
            "it_per_sec_final": logger.it_per_sec(),
            "samples_per_sec_final": logger.samples_per_sec(),
        })
        with open(os.path.join(outdir, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

# evaluate.py
# Minimal, single-file CLI for Chapter 5 evaluation & figures.
# Requires: torch, diffusers, open_clip_torch, geomloss, opencv-python, scikit-image, matplotlib, pandas, numpy, pillow

import os, time, math, json, argparse, itertools, shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

import torch
import torch.nn.functional as F

import cv2
from skimage.metrics import structural_similarity as ssim

# --- diffusion / adapters ---
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from ip_adapter_patch.degis_ip_adapter_patch import IPAdapter  # DEGIS implementation
from models.color_heads import ColorHead

# --- CLIP for CLIPScore ---
import open_clip

# --- EMD ---
try:
    from geomloss import SamplesLoss
    HAVE_GEOMLOSS = True
except Exception:
    HAVE_GEOMLOSS = False

# ---------------------------
# helpers: IO & palettes
# ---------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def load_manifest(manifest_csv):
    df = pd.read_csv(manifest_csv)
    # expect a column with local path
    if "file_path" not in df.columns and "local_path" in df.columns:
        df = df.rename(columns={"local_path": "file_path"})
    assert "file_path" in df.columns, "manifest must have a 'file_path' column"
    return df

def open_rgb(path, size=None):
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize((size, size), Image.BILINEAR)
    return img

# ---------------------------
# color histograms (match your training code)
# ---------------------------

def rgb_hist(img_np, bins=8):
    hist, _ = np.histogramdd(
        img_np.reshape(-1, 3), bins=(bins, bins, bins),
        range=((0,256),(0,256),(0,256))
    )
    h = hist.flatten().astype(np.float32)
    h /= (h.sum() + 1e-8)
    return h

def lab_hist(img_np, bins=8):
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab)
    L,a,b = cv2.split(lab)
    neutral_th = 5
    is_black = (L < 25) & (np.abs(a-128)<neutral_th) & (np.abs(b-128)<neutral_th)
    is_white = (L > 230) & (np.abs(a-128)<neutral_th) & (np.abs(b-128)<neutral_th)
    hist = cv2.calcHist([lab],[0,1,2],None,[bins,bins,bins],[0,256,0,256,0,256]).flatten()
    black_count = int(is_black.sum())
    white_count = int(is_white.sum())
    total = hist.sum() + black_count + white_count + 1e-8
    h = np.append(hist, [black_count, white_count]).astype(np.float32) / total
    return h

def hcl_hist(img_np, bins=8, c_max=150.0):
    # RGB -> Lab -> HCL
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2Lab).astype(np.float32)
    L,a8,b8 = cv2.split(lab)
    a = a8 - 128.0
    b = b8 - 128.0
    C = np.sqrt(a**2 + b**2)
    H = (np.degrees(np.arctan2(b,a)) + 360.0) % 360.0
    neutral_th = 5.0
    is_black = (L < 25) & (C < neutral_th)
    is_white = (L > 230) & (C < neutral_th)
    coords = np.stack([L.flatten(), C.flatten(), H.flatten()], axis=-1)
    hist, _ = np.histogramdd(coords, bins=(bins,bins,bins), range=((0,256),(0,c_max),(0,360)))
    hist = hist.flatten()
    black_count = int(is_black.sum()); white_count = int(is_white.sum())
    total = hist.sum() + black_count + white_count + 1e-8
    h = np.append(hist, [black_count, white_count]).astype(np.float32) / total
    return h

def compute_histogram(pil_img, color_space, bins=8):
    np_img = np.array(pil_img.resize((256,256)))
    if color_space == "rgb512":
        return rgb_hist(np_img, bins=bins)
    elif color_space == "lab514":
        return lab_hist(np_img, bins=bins)
    elif color_space == "hcl514":
        return hcl_hist(np_img, bins=bins)
    else:
        raise ValueError(f"unknown color_space {color_space}")

# ---------------------------
# metrics
# ---------------------------

def sinkhorn_emd(h1, h2, blur=0.05):
    """EMD between two 1D discrete measures laid on a uniform 1D grid of length D.
       This mirrors your training setup (vector-vs-vector with GeomLoss)."""
    if not HAVE_GEOMLOSS:
        # fallback: L1 as proxy (keeps script runnable if geomloss missing)
        return float(np.abs(h1 - h2).mean())
    D = h1.shape[0]
    x = torch.arange(D, dtype=torch.float32, device="cpu").view(D,1)  # 1D positions
    a = torch.tensor(h1, dtype=torch.float32).view(1, D)  # weights
    b = torch.tensor(h2, dtype=torch.float32).view(1, D)
    loss = SamplesLoss("sinkhorn", p=2, blur=blur, backend="tensorized")
    # GeomLoss expects samples; use positions x (shared) with weights a,b:
    return float(loss(x, x, a, b).item())

def neutral_bin_errors(h_gen, h_tgt, color_space):
    if color_space in ("lab514","hcl514"):
        black_err = float(abs(h_gen[-2] - h_tgt[-2]))
        white_err = float(abs(h_gen[-1] - h_tgt[-1]))
        return black_err, white_err
    return np.nan, np.nan

def edge_map(pil_img, size=512, low=100, high=200):
    img = np.array(pil_img.resize((size,size)).convert("L"))
    e = cv2.Canny(img, low, high)
    return (e > 0).astype(np.uint8)

def edge_f1(target_edges, pred_edges, tol=1):
    """F1 with tolerance by dilating target edges."""
    kernel = np.ones((2*tol+1,2*tol+1), np.uint8)
    target_dil = cv2.dilate(target_edges, kernel, iterations=1)
    tp = np.logical_and(pred_edges, target_dil).sum()
    fp = np.logical_and(pred_edges, ~target_dil).sum()
    fn = np.logical_and(~pred_edges, target_edges).sum()
    prec = tp / max(1, tp+fp); rec = tp / max(1, tp+fn)
    if prec+rec == 0: return 0.0, prec, rec
    f1 = 2*prec*rec/(prec+rec)
    return float(f1), float(prec), float(rec)

def edge_ssim(target_edges, pred_edges):
    # treat as binary images in [0,1]
    a = target_edges.astype(np.float32)
    b = pred_edges.astype(np.float32)
    return float(ssim(a, b, data_range=1.0))

def clip_score_openclip(model, preprocess, device, prompt, pil_img):
    img_t = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type=="cuda"):
        im = model.encode_image(img_t)
    im = F.normalize(im.float(), dim=-1)
    with torch.no_grad():
        txt = open_clip.tokenize([prompt]).to(device)
        with torch.cuda.amp.autocast(enabled=device.type=="cuda"):
            tt = model.encode_text(txt)
    tt = F.normalize(tt.float(), dim=-1)
    return float((im @ tt.T).squeeze().item())

# ---------------------------
# control image from manifest path
# ---------------------------

def control_edge_from_path(path, size=512):
    pil = Image.open(path).convert("RGB")
    gray = ImageOps.grayscale(pil)
    # auto-contrast helps ControlNet canny
    gray = ImageOps.autocontrast(gray)
    # convert to RGB (ControlNet expects 3-channel)
    return gray.resize((size,size), Image.BILINEAR).convert("RGB")

# ---------------------------
# montage figures
# ---------------------------

def palette_bar_figure(h, outpath, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,1.2))
    plt.bar(np.arange(len(h)), h)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()

def save_ablation_montage(imgs, labels, bars_pngs, outpath, cell=512, pad=12):
    """imgs: list[PIL.Image], labels: list[str], bars_pngs: list[path or None]"""
    cols = 2; rows = 2
    W = cols*cell + (cols+1)*pad
    H = rows*(cell+90) + (rows+1)*pad  # extra space under each for palette bar
    canvas = Image.new("RGB", (W,H), (245,245,245))
    draw = ImageDraw.Draw(canvas)
    for i,(im,lbl,barpng) in enumerate(zip(imgs, labels, bars_pngs)):
        r = i//cols; c = i%cols
        x = pad + c*(cell+pad); y = pad + r*(cell+90+pad)
        canvas.paste(im.resize((cell,cell), Image.LANCZOS), (x,y))
        draw.text((x, y+cell+4), lbl, fill=(0,0,0))
        if barpng and os.path.exists(barpng):
            bar = Image.open(barpng).convert("RGB").resize((cell,80), Image.BILINEAR)
            canvas.paste(bar, (x, y+cell+20))
    canvas.save(outpath)

# ---------------------------
# generation core
# ---------------------------

def build_pipelines(sd_id, controlnet_id, device, hf_cache=None):
    cn = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=torch.float16, cache_dir=hf_cache)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_id, controlnet=cn, torch_dtype=torch.float16,
        safety_checker=None, feature_extractor=None, cache_dir=hf_cache
    ).to(device)
    pipe.controlnet = pipe.controlnet.to(dtype=torch.float16)
    return pipe

def build_ip_adapter(pipe, ip_ckpt, image_encoder_path, device, num_tokens=4, embedding_type="custom"):
    return IPAdapter(
        sd_pipe=pipe,
        image_encoder_path=image_encoder_path,
        ip_ckpt=ip_ckpt,
        device=device,
        num_tokens=num_tokens,
        embedding_type=embedding_type,   # <-- use 'custom' so we can pass color head embeddings
    )

def build_color_head(ckpt_path, clip_dim, hist_dim, device):
    head = ColorHead(clip_dim=clip_dim, hist_dim=hist_dim).to(device).eval()
    head.load_state_dict(torch.load(ckpt_path, map_location=device))
    return head

def controlnet_knobs(scale):
    return dict(
        controlnet_conditioning_scale=scale,
        control_guidance_start=0.0,
        control_guidance_end=1.0,
    )

# ---------------------------
# main evaluation loop
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    # data & features
    ap.add_argument("--manifest_csv", required=True, help="CSV with file_path column (aligned with embeddings/hists by row index)")
    ap.add_argument("--embeddings_npy", required=True, help="np.loadable [N,1024] CLIP image embeddings (OpenCLIP ViT-H/14)")
    ap.add_argument("--hists_npy", required=True, help="np.loadable [N, hist_dim] histograms in the chosen color space")
    ap.add_argument("--color_space", choices=["rgb512","lab514","hcl514"], default="hcl514")
    ap.add_argument("--color_head_ckpt", required=True)
    ap.add_argument("--indices", type=str, default=None, help="comma-separated indices to evaluate (optional)")
    ap.add_argument("--prompts_csv", required=True, help="CSV with columns: idx (int matching manifest row), prompt (str)")
    # generation stack
    ap.add_argument("--sd_id", default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--controlnet_id", default="lllyasviel/control_v11p_sd15_canny")
    ap.add_argument("--ip_ckpt", required=True, help="path to ip-adapter_sd15.bin or .safetensors")
    ap.add_argument("--image_encoder_path", default="laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    ap.add_argument("--hf_cache", default=None)
    # sweeps
    ap.add_argument("--ip_scales", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument("--cn_scales", type=str, default="0.8,1.2,1.6,2.0")
    ap.add_argument("--cfg", type=float, default=7.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=123)
    # edges
    ap.add_argument("--canny_low", type=int, default=100)
    ap.add_argument("--canny_high", type=int, default=200)
    ap.add_argument("--edge_size", type=int, default=512)
    # output
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max rows; 0 = all in prompts_csv")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.outdir)
    img_dir = Path(args.outdir) / "imgs"; ensure_dir(img_dir)
    fig_dir = Path(args.outdir) / "figs"; ensure_dir(fig_dir)
    csv_dir = Path(args.outdir) / "csv"; ensure_dir(csv_dir)

    # data
    df_manifest = load_manifest(args.manifest_csv)
    df_prompts = pd.read_csv(args.prompts_csv)
    if args.limit and len(df_prompts) > args.limit:
        df_prompts = df_prompts.head(args.limit).copy()

    emb = np.load(args.embeddings_npy, mmap_mode="r")
    hists = np.load(args.hists_npy, mmap_mode="r")
    N, clip_dim = emb.shape
    hist_dim = hists.shape[1]
    assert N == len(df_manifest), "embeddings/hists must align with manifest rows"

    # models
    pipe = build_pipelines(args.sd_id, args.controlnet_id, device, hf_cache=args.hf_cache)
    ip_adapter = build_ip_adapter(pipe, args.ip_ckpt, args.image_encoder_path, device, embedding_type="custom")
    color_head = build_color_head(args.color_head_ckpt, clip_dim=clip_dim, hist_dim=hist_dim, device=device)

    # CLIPScore model
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=args.hf_cache
    )
    clip_model = clip_model.to(device)
    if device.type == "cuda":
        clip_model = clip_model.half()
    clip_model.eval()

    ip_scales = [float(x) for x in args.ip_scales.split(",")]
    cn_scales = [float(x) for x in args.cn_scales.split(",")]

    rows = []
    # --- loop prompts
    for ridx, row in df_prompts.iterrows():
        idx = int(row["idx"])
        prompt = str(row["prompt"])

        # target palette + control edges from the SAME source frame
        src_path = df_manifest.iloc[idx]["file_path"]
        tgt_hist = hists[idx].astype(np.float32)
        # color head embedding from CLIP emb
        z = torch.tensor(emb[idx], dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            _, _, c_emb = color_head(z)
        c_emb = c_emb.float()  # [1,1024]
        control_pil = control_edge_from_path(src_path, size=args.edge_size)

        # --- 2x2 ablation (single default scales for figure)
        ablation = [
            ("Text only", 0.0, None),
            ("+Edges", 0.0, 1.6),
            ("+Colour", 0.75, None),
            ("+Both", 0.75, 1.6),
        ]
        ablation_imgs, ablation_labels, ablation_bars = [], [], []
        for label, ip_s, cn_s in ablation:
            ip_adapter.set_scale(ip_s)
            kw = {}
            if cn_s is not None:
                kw.update(dict(image=control_pil))
                kw.update(controlnet_knobs(cn_s))
            t0 = time.time()
            imgs = ip_adapter.generate_from_embeddings(
                clip_image_embeds=c_emb.half(),  # pass color embedding
                prompt=prompt, negative_prompt=None,
                scale=ip_s, num_samples=1, seed=args.seed,
                guidance_scale=args.cfg, num_inference_steps=args.steps,
                **kw
            )
            latency = time.time() - t0
            gen = imgs[0].convert("RGB")
            # metrics
            gen_hist = compute_histogram(gen, args.color_space)
            emd = sinkhorn_emd(gen_hist, tgt_hist)
            b_err, w_err = neutral_bin_errors(gen_hist, tgt_hist, args.color_space)
            # edges
            pred_edges = edge_map(gen, size=args.edge_size, low=args.canny_low, high=args.canny_high)
            targ_edges = edge_map(control_pil, size=args.edge_size, low=args.canny_low, high=args.canny_high)
            f1, prec, rec = edge_f1(targ_edges, pred_edges, tol=1)
            ess = edge_ssim(targ_edges, pred_edges)
            cs = clip_score_openclip(clip_model, clip_preprocess, device, prompt, gen)

            # save image
            out_name = f"{ridx:03d}_{label.replace('+','plus').replace(' ','_')}.png"
            gen_path = img_dir / out_name
            gen.save(gen_path)

            # save bar figures
            bar_tgt = fig_dir / f"{ridx:03d}_tgtbar.png"
            bar_gen = fig_dir / f"{ridx:03d}_{label.replace('+','plus').replace(' ','_')}_bar.png"
            if not os.path.exists(bar_tgt):
                palette_bar_figure(tgt_hist, bar_tgt, title="Target palette")
            palette_bar_figure(gen_hist, bar_gen, title="Generated palette")

            # store for montage
            ablation_imgs.append(gen)
            ablation_labels.append(f"{label}\nEMD={emd:.4f}  F1={f1:.3f}  CLIP={cs:.3f}")
            ablation_bars.append(str(bar_gen))

            # csv row
            rows.append(dict(
                prompt_idx=idx, row_id=ridx, mode=label, ip_scale=ip_s,
                cn_scale=(cn_s if cn_s is not None else 0.0),
                cfg=args.cfg, steps=args.steps, latency_s=latency,
                color_emd=emd, neutral_black_err=b_err, neutral_white_err=w_err,
                edge_f1=f1, edge_prec=prec, edge_rec=rec, edge_ssim=ess,
                clip_score=cs, image_path=str(gen_path)
            ))

        # save 2x2 montage
        montage_path = fig_dir / f"{ridx:03d}_ablation_montage.png"
        save_ablation_montage(ablation_imgs, ablation_labels,
                              [None] + ablation_bars[1:],  # put bars for 3 generated, target bar not needed here
                              montage_path)

        # --- scale grid sweep
        for ip_s, cn_s in itertools.product(ip_scales, cn_scales):
            ip_adapter.set_scale(ip_s)
            kw = {}
            kw.update(dict(image=control_pil))
            kw.update(controlnet_knobs(cn_s))
            t0 = time.time()
            imgs = ip_adapter.generate_from_embeddings(
                clip_image_embeds=c_emb.half(),
                prompt=prompt, negative_prompt=None,
                scale=ip_s, num_samples=1, seed=args.seed,
                guidance_scale=args.cfg, num_inference_steps=args.steps,
                **kw
            )
            latency = time.time() - t0
            gen = imgs[0].convert("RGB")

            gen_hist = compute_histogram(gen, args.color_space)
            emd = sinkhorn_emd(gen_hist, tgt_hist)
            b_err, w_err = neutral_bin_errors(gen_hist, tgt_hist, args.color_space)
            pred_edges = edge_map(gen, size=args.edge_size, low=args.canny_low, high=args.canny_high)
            targ_edges = edge_map(control_pil, size=args.edge_size, low=args.canny_low, high=args.canny_high)
            f1, prec, rec = edge_f1(targ_edges, pred_edges, tol=1)
            ess = edge_ssim(targ_edges, pred_edges)
            cs = clip_score_openclip(clip_model, clip_preprocess, device, prompt, gen)

            out_name = f"{ridx:03d}_grid_ip{ip_s}_cn{cn_s}.png".replace(".","p")
            gen_path = img_dir / out_name
            gen.save(gen_path)

            rows.append(dict(
                prompt_idx=idx, row_id=ridx, mode="grid",
                ip_scale=ip_s, cn_scale=cn_s, cfg=args.cfg, steps=args.steps,
                latency_s=latency, color_emd=emd,
                neutral_black_err=b_err, neutral_white_err=w_err,
                edge_f1=f1, edge_prec=prec, edge_rec=rec, edge_ssim=ess,
                clip_score=cs, image_path=str(gen_path)
            ))

    # write CSV
    out_csv = csv_dir / "metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n✅ Done. Metrics at: {out_csv}")
    print(f"   Images: {img_dir}")
    print(f"   Figures: {fig_dir}")

if __name__ == "__main__":
    main()

# save as summarize_runs.py and run:  python summarize_runs.py
import json, pandas as pd, glob, os

def summarize(run):
    meta = json.load(open(os.path.join(run, "run_meta.json")))
    df   = pd.read_csv(os.path.join(run, "metrics.csv"))
    best_idx = df['val_emd'].idxmin()
    best = df.loc[best_idx]

    ck_color = sorted(glob.glob(os.path.join(run, "..", "best_color_head*.pth")))
    ck_rest  = sorted(glob.glob(os.path.join(run, "..", "best_rest_head*.pth")))

    out = {
        "run_dir": run,
        "run_name": meta.get("run_name"),
        "hist_kind": meta.get("hist_kind"),
        "top_k": meta.get("top_k"),
        "blur": meta.get("blur"),
        "lambda_ortho": meta.get("lambda_ortho"),
        "lambda_leak": meta.get("lambda_leak"),
        "epochs": meta.get("epochs"),
        "batch_size": meta.get("batch_size"),
        "val_batch_size": meta.get("val_batch_size"),
        "lr": meta.get("lr"),
        "weight_decay": meta.get("weight_decay"),
        "device": meta.get("device"),
        "gpu_name": meta.get("gpu_name"),
        "param_count_color": meta.get("param_count_color"),
        "param_count_rest": meta.get("param_count_rest"),
        "git_commit": meta.get("git_commit"),
        "wall_seconds": meta.get("wall_seconds"),
        "it_per_sec_final": meta.get("it_per_sec_final"),
        "samples_per_sec_final": meta.get("samples_per_sec_final"),
        # from metrics.csv
        "best_epoch": int(best['epoch']),
        "best_val_emd": float(best['val_emd']),
        "train_emd_at_best": float(best['train_emd']),
        "diag_bce_at_best": float(best['diag_bce']),
        # checkpoints
        "checkpoint_color": ck_color[-1] if ck_color else "best_color_head.pth",
        "checkpoint_rest":  ck_rest[-1]  if ck_rest  else "best_rest_head.pth",
    }
    return out

runs = sorted([p for p in glob.glob("runs/*") if os.path.isdir(p)])
for r in runs:
    try:
        s = summarize(r)
        print("\n=== paste this block ===")
        for k,v in s.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"[skip] {r}: {e}")

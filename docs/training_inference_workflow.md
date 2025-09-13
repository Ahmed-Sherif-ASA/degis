sequenceDiagram
    participant IN as Inputs (CSV with URLs, Local Images Path)
    participant Train as Training Module
    participant Models as Trained Models
    participant Infer as Inference Module
    participant OUT as Outputs (images + metrics)

    Note over IN,OUT: TRAINING PHASE
    IN->>Train: Load dataset images (from CSV/URLs)
    par Compute CLIP embeddings
        Train->>Train: Compute CLIP embeddings
    and Compute colour histograms
        Train->>Train: Compute colour histograms (RGB/LAB/HCL)
    end
    Train->>Models: Save checkpoints (Colour Head)
    %% Optional ack to be explicit
    Models-->>Train: Saved (path/id)

    Note over Models: Trained once per encoder–IP-Adapter–SD version

    Note over IN,OUT: INFERENCE PHASE
    IN->>Infer: Load text prompt · colour reference · layout reference
    Infer->>Models: Load Colour Head + encoder versions
    Models-->>Infer: Return weights + config (ckpt, encoder_id)
    Infer->>Infer: Encode colour reference (CLIP)
    Infer->>Infer: Predict colour embedding (Colour Head)
    Infer->>Infer: Create Canny control image from layout reference
    Infer->>Infer: IP-Adapter token concat + scaling\n(attn_ip_scale, text_token_scale, ip_token_scale)
    loop EMD-constrained outer loop (until EMD ≤ τ or max attempts)
        Infer->>Infer: Generate image with UNet + ControlNet
        Infer->>Infer: Compute EMD to target palette
    end
    Infer->>OUT: Save image(s), EMD, settings, logs

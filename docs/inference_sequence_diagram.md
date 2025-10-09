```mermaid
sequenceDiagram
    %% PARTICIPANTS
    participant IN as Inputs (prompt · color ref · layout ref)
    participant Models as Trained Models
    participant Infer as Inference Module
    participant OUT as Outputs (images + metrics)

    Note over IN,OUT: INFERENCE PHASE
    IN->>Infer: Load text prompt · color reference · layout reference
    Infer->>Models: Load Color Head + encoder versions
    Models-->>Infer: Return weights + config (ckpt, encoder_id, ip_adapter_id, sd_version)

    Infer->>Infer: Encode color reference (CLIP)
    Infer->>Infer: Predict palette embedding (Color Head)

    Infer->>Infer: Create Canny control image from layout reference

    Infer->>Infer: IP-Adapter token concat + scaling\n(attn_ip_scale, text_token_scale, ip_token_scale)

    loop Sinkhorn-constrained outer loop (until Sinkhorn ≤ τ or max attempts)
        Infer->>Infer: Generate image with UNet + ControlNet
        Infer->>Infer: Compute Sinkhorn to target palette
    end

    Infer->>OUT: Save image(s), Sinkhorn, settings, logs

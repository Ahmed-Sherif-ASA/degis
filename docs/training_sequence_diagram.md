```mermaid
sequenceDiagram
    %% PARTICIPANTS
    participant IN as Inputs (CSV/URLs)
    participant Train as Training Module
    participant Models as Trained Models

    Note over IN,Models: TRAINING PHASE
    IN->>Train: Load dataset images
    par Compute CLIP embeddings
        Train->>Train: Compute CLIP embeddings
    and Compute color histograms
        Train->>Train: Compute color histograms (RGB/LAB/HCL)
    end
    Train->>Models: Train color Head (CLIP → palette embedding) and save checkpoint
    Models-->>Train: Saved (ckpt path · encoder_id · ip_adapter_id · sd_version)
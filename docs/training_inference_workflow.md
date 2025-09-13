# Training vs Inference Workflow

This diagram shows the complete workflow from training to inference in the DEGIS architecture.

```mermaid
sequenceDiagram
    participant CSV as CSV Data
    participant Train as Training Module
    participant Models as Trained Models
    participant Infer as Inference Module
    participant Output as Generated Images
    
    Note over CSV,Output: TRAINING PHASE
    CSV->>Train: Load Images
    Train->>Train: Generate CLIP Embeddings
    Train->>Train: Extract Color Histograms
    Train->>Models: Train Color Disentanglement Models
    Models->>Models: Save Checkpoints
    
    Note over CSV,Output: INFERENCE PHASE
    CSV->>Infer: Load Images
    Infer->>Models: Load Trained Color Models
    Infer->>Infer: Generate Color Embeddings
    Infer->>Infer: Create Control Images from Edge Maps
    Infer->>Output: Generate Images with IP-Adapter + ControlNet
    Infer->>Output: Apply EMD Constraints
```

## Key Points

- **Training Phase**: Focuses on learning color disentanglement from CLIP embeddings and color histograms
- **Inference Phase**: Uses trained models to generate images with IP-Adapter and ControlNet
- **Clean Separation**: Training and inference modules are completely independent
- **Shared Components**: Common utilities (config, visualization, etc.) are in the `shared/` folder

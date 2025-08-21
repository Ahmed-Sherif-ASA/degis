import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"

clip_model, _, preprocess = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained,
    device=device
)
clip_model = clip_model.to(device).half()
clip_model.eval()



def compute_clip_embedding(image_tensor: torch.Tensor) -> torch.Tensor:
    image_input = image_tensor.unsqueeze(0).to(device)  # [1,3,H,W]
    with torch.no_grad():
        z = clip_model.encode_image(image_input).float().squeeze(0)
    return z

def generate_embeddings(loader, embeddings_path, compute_clip_embedding, force_recompute=False):
    print("### CLIP Embedding Generation ###")
    print("Total images:", len(loader.dataset))

    try:
        if not force_recompute:
            embeddings = np.load(embeddings_path)
            print(f"Loaded from {embeddings_path}")
            return embeddings
    except Exception as e:
        print(f"Loading failed: {e}. Recomputing...")

    emb_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing CLIP embeddings"):
            imgs, _ = batch
            for img in imgs:
                z = compute_clip_embedding(img)
                emb_list.append(z.cpu().numpy())

    embeddings = np.stack(emb_list, axis=0)
    np.save(embeddings_path, embeddings)
    print(f"Saved to {embeddings_path}")
    return embeddings

def generate_embeddings_fp16(loader, save_path, force_recompute=False):
    try:
        if not force_recompute:
            emb = np.load(save_path)
            print(f"→ Loaded precomputed from {save_path}")
            return emb
    except FileNotFoundError:
        print("→ No existing file, recomputing…")

    all_embs = []
    for imgs, _ in tqdm(loader, desc="CLIP batched encode"):
        # imgs: [B,3,H,W] in float32 [0,1]
        imgs = imgs.to(device).half()                 # to fp16
        with torch.no_grad(), torch.cuda.amp.autocast():
            z = clip_model.encode_image(imgs)         # [B,1024] fp16
        all_embs.append(z.float().cpu().numpy())      # back to fp32 on CPU

    embeddings = np.concatenate(all_embs, axis=0)
    np.save(save_path, embeddings)
    print(f"→ Saved embeddings to {save_path}")
    return embeddings



embeddings = generate_embeddings_fp16(
    loader,
    'test_embedding.npy',
    force_recompute=True
)
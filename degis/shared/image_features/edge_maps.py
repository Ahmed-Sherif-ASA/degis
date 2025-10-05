import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def compute_edge_map_canny(img: Image.Image,
                           resize_size=(224, 224)) -> np.ndarray:
    img_gray = np.array(img.resize(resize_size).convert("L"))
    edges    = cv.Canny(img_gray, 150, 200) / 255.0
    return edges.astype(np.float32).flatten()

def generate_edge_maps(loader, edge_maps_path, method="canny", resize_size=(224,224), force_recompute=False):
    num_images = len(loader.dataset)
    edge_dim   = resize_size[0] * resize_size[1]

    if not force_recompute:
        try:
            edge_maps = np.load(edge_maps_path)
            print(f"[OK] Loaded cached edge maps from {edge_maps_path}")
            return edge_maps
        except Exception:
            print("[!] Cache not found -> recompute")

    edge_maps = np.zeros((num_images, edge_dim), dtype=np.uint8)
    print(f"### Edge-map generation using {method.upper()} ###")

    idx = 0
    for batch in tqdm(loader, desc="Generating edge maps"):
        imgs, *_ = batch
        for img_tensor in imgs:
            img_pil = transforms.ToPILImage()(img_tensor)

            if method == "canny":
                edge_np = compute_edge_map_canny(img_pil, resize_size)
            else:
                raise ValueError("method must be 'canny'")

            edge_maps[idx] = edge_np
            idx += 1

    np.save(edge_maps_path, edge_maps)
    print(f"[OK] Edge maps saved to {edge_maps_path}")
    return edge_maps


import cv2 as cv
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ───────────────────  Register the missing Crop layer  ─────────────────
# (Only needed for HED, safe to keep or remove)
# class CropLayer(cv.dnn.Layer):
#     def __init__(self, params, blobs):
#         super().__init__()
#         self.x0 = self.y0 = self.x1 = self.y1 = 0
#
#     def getMemoryShapes(self, inputs):
#         inShape, targetShape = inputs[0], inputs[1]   # (N,C,H,W)
#         N, C = inShape[0], inShape[1]
#         Ht, Wt = targetShape[2], targetShape[3]
#
#         self.y0 = (inShape[2] - Ht) // 2
#         self.x0 = (inShape[3] - Wt) // 2
#         self.y1 = self.y0 + Ht
#         self.x1 = self.x0 + Wt
#         return [[N, C, Ht, Wt]]
#
#     def forward(self, inputs):
#         return [inputs[0][:, :, self.y0:self.y1, self.x0:self.x1]]
#
# cv.dnn_registerLayer("Crop", CropLayer)

# ───────────────────────────  Load HED model  ──────────────────────────
# (Commented out since you don’t need HED now)
# hed_proto = "/data/thesis/models/deploy.prototxt"
# hed_model = "/data/thesis/models/hed_pretrained_bsds.caffemodel"
# hed_net   = cv.dnn.readNetFromCaffe(hed_proto, hed_model)

# ─────────────────────  Edge-map helper functions  ─────────────────────
def compute_edge_map_canny(img: Image.Image,
                           resize_size=(224, 224)) -> np.ndarray:
    img_gray = np.array(img.resize(resize_size).convert("L"))
    edges    = cv.Canny(img_gray, 150, 200) / 255.0
    return edges.astype(np.float32).flatten()

# def compute_edge_map_hed(img: Image.Image,
#                          resize_size=(224, 224)) -> np.ndarray:
#     img_bgr = cv.cvtColor(np.array(img.resize(resize_size)), cv.COLOR_RGB2BGR)
#     blob = cv.dnn.blobFromImage(
#         img_bgr,
#         scalefactor=1.0,
#         size=resize_size,
#         mean=(104.00698793, 116.66876762, 122.67891434),
#         swapRB=False,
#         crop=False
#     )
#     hed_net.setInput(blob)
#     hed = hed_net.forward()[0, 0]
#     hed_norm = hed.clip(0, 255) / 255.0
#     return hed_norm.astype(np.float32).flatten()

# ─────────────────────  Batch generator (PyTorch)  ─────────────────────
from tqdm import tqdm

def generate_edge_maps(loader, edge_maps_path, method="canny", resize_size=(224,224), force_recompute=False):
    num_images = len(loader.dataset)
    edge_dim   = resize_size[0] * resize_size[1]

    if not force_recompute:
        try:
            edge_maps = np.load(edge_maps_path)
            print(f"[✓] Loaded cached edge maps from {edge_maps_path}")
            return edge_maps
        except Exception:
            print("[!] Cache not found -> recompute")

    # edge_maps = np.zeros((num_images, edge_dim), dtype=np.float32)
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
    print(f"[✓] Edge maps saved to {edge_maps_path}")
    return edge_maps


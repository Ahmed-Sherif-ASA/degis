import numpy as np
import torch
from tqdm import tqdm
from skimage.color import rgb2lab
import cv2
import math
from PIL import Image
from torchvision import transforms

def fast_rgb_histogram(img_tensor: torch.Tensor, bins: int = 8) -> np.ndarray:
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255.0).astype(np.uint8)
    hist, _ = np.histogramdd(
        img.reshape(-1, 3),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256))
    )
    hist = hist.flatten()
    hist /= hist.sum() + 1e-8
    return hist


def compute_color_histogram(img: Image.Image, bins: int = 8) -> np.ndarray:
    img_resized = img.resize((256, 256))
    img_np = np.array(img_resized)
    hist = cv2.calcHist([img_np], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    hist /= hist.sum() + 1e-8
    return hist


def fast_lab_histogram(img_tensor: torch.Tensor, bins: int = 8) -> np.ndarray:
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    lab_float = rgb2lab(img_np)
    L, a, b = lab_float[..., 0], lab_float[..., 1], lab_float[..., 2]
    neutral_threshold = 2.0
    is_black = (L < 10) & (np.abs(a) < neutral_threshold) & (np.abs(b) < neutral_threshold)
    is_white = (L > 90) & (np.abs(a) < neutral_threshold) & (np.abs(b) < neutral_threshold)
    L8 = (L * 255 / 100).clip(0, 255).astype(np.uint8)
    a8 = (a + 128).clip(0, 255).astype(np.uint8)
    b8 = (b + 128).clip(0, 255).astype(np.uint8)
    lab8 = np.stack([L8, a8, b8], axis=-1)
    hist, _ = np.histogramdd(
        lab8.reshape(-1, 3),
        bins=(bins, bins, bins),
        range=((0, 256), (0, 256), (0, 256))
    )
    hist = hist.flatten()
    black_count = np.sum(is_black)
    white_count = np.sum(is_white)
    total = hist.sum() + black_count + white_count + 1e-8
    hist = np.append(hist, [black_count, white_count]) / total
    return hist


def compute_lab_histogram(img: Image.Image, bins: int = 8) -> np.ndarray:
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(img_lab)
    neutral_threshold = 5
    is_black = (L < 25) & (np.abs(a - 128) < neutral_threshold) & (np.abs(b - 128) < neutral_threshold)
    is_white = (L > 230) & (np.abs(a - 128) < neutral_threshold) & (np.abs(b - 128) < neutral_threshold)
    hist = cv2.calcHist([img_lab], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    hist = hist.flatten()
    black_count = np.sum(is_black)
    white_count = np.sum(is_white)
    total = hist.sum() + black_count + white_count + 1e-8
    hist = np.append(hist, [black_count, white_count]) / total
    return hist


def fast_hcl_histogram(img_tensor: torch.Tensor, bins: int = 8, c_max: float = 150.0) -> np.ndarray:
    """
    Compute a global HCL histogram directly from a [3, H, W] tensor.
    Returns normalized histogram of shape (bins^3 + 2,).
    """
    img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
    lab = rgb2lab(img_np)
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a**2 + b**2)
    H = np.degrees(np.arctan2(b, a)) % 360
    # Black/white masks
    neutral_thresh = 2.0
    is_black = (L < 10) & (C < neutral_thresh)
    is_white = (L > 90) & (C < neutral_thresh)
    # Prepare data for histogram
    coords = np.stack([L.flatten(), C.flatten(), H.flatten()], axis=-1)
    hist, _ = np.histogramdd(
        coords,
        bins=(bins, bins, bins),
        range=((0, 100), (0, c_max), (0, 360))
    )
    hist = hist.flatten()
    black_count = is_black.sum()
    white_count = is_white.sum()
    total = hist.sum() + black_count + white_count + 1e-8
    hist = np.append(hist, [black_count, white_count]) / total
    return hist


def compute_hcl_histogram(img: Image.Image, bins: int = 8, c_max: float = 150.0) -> np.ndarray:
    """OpenCV-based HCL histogram with black/white bins"""
    img_np = np.array(img)
    lab_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(lab_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
    L, a8, b8 = cv2.split(lab)
    a = a8 - 128.0
    b = b8 - 128.0
    C = np.sqrt(a**2 + b**2)
    H = (np.degrees(np.arctan2(b, a)) + 360) % 360
    neutral_thresh = 5.0
    is_black = (L < 25) & (C < neutral_thresh)
    is_white = (L > 230) & (C < neutral_thresh)
    coords = np.stack([L.flatten(), C.flatten(), H.flatten()], axis=-1)
    hist, _ = np.histogramdd(
        coords,
        bins=(bins, bins, bins),
        range=((0, 256), (0, c_max), (0, 360))
    )
    hist = hist.flatten()
    black_count = int(is_black.sum())
    white_count = int(is_white.sum())
    total = hist.sum() + black_count + white_count + 1e-8
    hist = np.append(hist, [black_count, white_count]) / total
    return hist


def generate_color_histograms(
    loader,
    hist_path,
    force_recompute=False,
    hist_bins=8,
    method="fast",
    color_space="rgb"
):
    print(f"### Color Histogram Generation [{method.upper()}, {color_space.upper()}] ###")
    num_images = len(loader.dataset)
    # Determine histogram dimension
    hist_dim = hist_bins ** 3
    if color_space in ("lab", "hcl"):
        hist_dim += 2
    print(f"Total images: {num_images}, Bins: {hist_bins}, Dimensions: {hist_dim}")

    if not force_recompute:
        try:
            histograms = np.load(hist_path)
            print(f"Loaded from {hist_path}")
            return histograms
        except Exception as e:
            print(f"Loading failed. Recomputing...")

    all_histograms = np.zeros((num_images, hist_dim), dtype=np.float32)

    for i in tqdm(range(num_images), desc=f"Computing histograms [{method}, {color_space}]"):
        img_tensor, _ = loader.dataset[i]

        if color_space == "rgb":
            if method == "fast":
                hist_np = fast_rgb_histogram(img_tensor, bins=hist_bins)
            else:
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                hist_np = compute_color_histogram(img_pil, bins=hist_bins)

        elif color_space == "lab":
            if method == "fast":
                hist_np = fast_lab_histogram(img_tensor, bins=hist_bins)
            else:
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                hist_np = compute_lab_histogram(img_pil, bins=hist_bins)

        elif color_space == "hcl":
            if method == "fast":
                hist_np = fast_hcl_histogram(img_tensor, bins=hist_bins)
            else:
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                hist_np = compute_hcl_histogram(img_pil, bins=hist_bins)

        else:
            raise ValueError(f"Invalid color space: {color_space}. Use 'rgb', 'lab', or 'hcl'.")

        all_histograms[i] = hist_np

    np.save(hist_path, all_histograms)
    print(f"Saved to {hist_path}")
    return all_histograms
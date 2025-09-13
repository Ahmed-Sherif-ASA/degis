"""
Visualization utilities for generated images and color palettes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Tuple, Optional
from skimage.color import lab2rgb


def plot_color_palette(
    colors: List[Tuple[float, float, float]], 
    values: List[float],
    title: str = "Color Palette",
    figsize: Tuple[int, int] = (10, 2)
) -> plt.Figure:
    """Plot a color palette with values."""
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    for i, (c, v) in enumerate(zip(colors, values)):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        ax.text(i+0.5, -0.08, f"{v:.3f}", ha="center", va="top", fontsize=7)
    
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=12)
    
    return fig


def display_images_grid(
    images: List[Image.Image],
    cols: int = 3,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 512
) -> Image.Image:
    """Display multiple images in a grid layout."""
    if not images:
        return None
    
    # Resize all images to the same size
    resized_images = [img.resize((size, size)) for img in images]
    
    w, h = size, size
    rows = (len(resized_images) + cols - 1) // cols
    
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding
    
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)
    
    for i, img in enumerate(resized_images):
        row, col = divmod(i, cols)
        x = col * (w + padding)
        y = row * (h + padding)
        grid_img.paste(img, (x, y))
    
    return grid_img


def display_comparison_grid(
    original: Image.Image,
    control: Image.Image,
    generated: List[Image.Image],
    cols: int = 3,
    padding: int = 10,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    size: int = 512
) -> Image.Image:
    """Display original, control, and generated images in a comparison grid."""
    all_images = [original, control] + generated
    return display_images_grid(all_images, cols, padding, bg_color, size)


def extract_top_palette(
    histogram: np.ndarray,
    bins: int = 8,
    top_k: int = 20,
    color_space: str = "rgb",
    c_max: float = 150.0
) -> Tuple[List[Tuple[float, float, float]], np.ndarray]:
    """Extract top-k colors from a histogram."""
    core = histogram[:bins**3]  # ignore black/white bins
    idxs = np.argsort(core)[-top_k:][::-1]
    colors = []
    
    if color_space == "rgb":
        for flat in idxs:
            ri = flat // (bins * bins)
            gi = (flat // bins) % bins
            bi = flat % bins
            colors.append(((ri + 0.5) / bins, (gi + 0.5) / bins, (bi + 0.5) / bins))
    
    elif color_space == "lab":
        for flat in idxs:
            Li = flat // (bins * bins)
            ai = (flat // bins) % bins
            bi = flat % bins
            L = (Li + 0.5) / bins * 100.0
            a = (ai + 0.5) / bins * 255.0 - 128.0
            b = (bi + 0.5) / bins * 255.0 - 128.0
            colors.append(tuple(lab2rgb(np.array([[[L, a, b]]]))[0, 0]))
    
    elif color_space == "hcl":
        for flat in idxs:
            Li = flat // (bins * bins)
            Ci = (flat // bins) % bins
            Hi = flat % bins
            L = (Li + 0.5) / bins * 100.0
            C = (Ci + 0.5) / bins * c_max
            H = (Hi + 0.5) / bins * 360.0
            a = C * np.cos(np.deg2rad(H))
            b = C * np.sin(np.deg2rad(H))
            colors.append(tuple(lab2rgb(np.array([[[L, a, b]]]))[0, 0]))
    
    else:
        raise ValueError("color_space must be 'rgb', 'lab', or 'hcl'")
    
    return colors, core[idxs]


def visualize_histogram_comparison(
    gt_histogram: np.ndarray,
    pred_histogram: np.ndarray,
    bins: int = 8,
    top_k: int = 20,
    color_space: str = "rgb",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """Visualize ground truth vs predicted histogram comparison."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Extract top palettes
    gt_colors, gt_values = extract_top_palette(gt_histogram, bins, top_k, color_space)
    pred_colors, pred_values = extract_top_palette(pred_histogram, bins, top_k, color_space)
    
    # Plot ground truth
    for i, (c, v) in enumerate(zip(gt_colors, gt_values)):
        axes[0].add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        axes[0].text(i+0.5, -0.08, f"{v:.3f}", ha="center", va="top", fontsize=7)
    axes[0].set_xlim(0, len(gt_colors))
    axes[0].set_ylim(0, 1)
    axes[0].axis("off")
    axes[0].set_title(f"Ground Truth top-{top_k}", fontsize=12)
    
    # Plot prediction
    for i, (c, v) in enumerate(zip(pred_colors, pred_values)):
        axes[1].add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
        axes[1].text(i+0.5, -0.08, f"{v:.3f}", ha="center", va="top", fontsize=7)
    axes[1].set_xlim(0, len(pred_colors))
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].set_title(f"Prediction top-{top_k}", fontsize=12)
    
    plt.tight_layout()
    return fig


def plot_training_metrics(
    metrics: dict,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """Plot training metrics over time."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot EMD losses
    if 'train_emd' in metrics and 'val_emd' in metrics:
        axes[0, 0].plot(metrics['train_emd'], label='Train EMD')
        axes[0, 0].plot(metrics['val_emd'], label='Val EMD')
        axes[0, 0].set_title('EMD Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('EMD')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
    
    # Plot BCE loss
    if 'diag_bce' in metrics:
        axes[0, 1].plot(metrics['diag_bce'], label='BCE')
        axes[0, 1].set_title('Binary Cross Entropy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('BCE')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Plot learning rate
    if 'lr' in metrics:
        axes[1, 0].plot(metrics['lr'], label='Learning Rate')
        axes[1, 0].set_title('Learning Rate')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LR')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot gradient norm
    if 'grad_norm' in metrics:
        axes[1, 1].plot(metrics['grad_norm'], label='Grad Norm')
        axes[1, 1].set_title('Gradient Norm')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Norm')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig


def save_generation_results(
    images: List[Image.Image],
    output_dir: str,
    prefix: str = "generated",
    format: str = "PNG"
):
    """Save generated images to disk."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for i, img in enumerate(images):
        filename = f"{prefix}_{i:03d}.{format.lower()}"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath, format)
    
    print(f"Saved {len(images)} images to {output_dir}")


def create_side_by_side_comparison(
    original: Image.Image,
    generated: List[Image.Image],
    titles: List[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """Create a side-by-side comparison of original and generated images."""
    n_images = len(generated) + 1  # +1 for original
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    # Plot original
    axes[0].imshow(original)
    axes[0].axis('off')
    axes[0].set_title(titles[0] if titles else "Original")
    
    # Plot generated images
    for i, img in enumerate(generated):
        axes[i + 1].imshow(img)
        axes[i + 1].axis('off')
        title = titles[i + 1] if titles and i + 1 < len(titles) else f"Generated {i + 1}"
        axes[i + 1].set_title(title)
    
    plt.tight_layout()
    return fig


def plot_training_curves(
    metrics_csv_path: str,
    output_dir: str,
    dataset_name: str,
    hist_kind: str
) -> None:
    """
    Generate training curves (EMD and loss) from metrics CSV.
    
    Args:
        metrics_csv_path: Path to metrics.csv file
        output_dir: Directory to save plots
        dataset_name: Name of dataset for plot titles
        hist_kind: Type of histogram for plot titles
    """
    try:
        import pandas as pd
        import os
        
        df = pd.read_csv(metrics_csv_path)
        
        # EMD curves
        plt.figure(figsize=(10, 6))
        plt.plot(df.epoch, df.train_emd, label="train", linewidth=2)
        plt.plot(df.epoch, df.val_emd, label="val", linewidth=2)
        plt.xlabel("epoch")
        plt.ylabel("EMD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"EMD Curves - {dataset_name}_{hist_kind}")
        plt.savefig(os.path.join(output_dir, "emd_curves.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Loss curve
        plt.figure(figsize=(10, 6))
        plt.plot(df.epoch, df["loss"], linewidth=2, color='red')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.grid(True, alpha=0.3)
        plt.title(f"Loss Curve - {dataset_name}_{hist_kind}")
        plt.savefig(os.path.join(output_dir, "loss_curve.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Generated training curves (emd_curves.png, loss_curve.png)")
        
    except ImportError:
        print("⚠️  matplotlib not available, skipping training curves")
    except Exception as e:
        print(f"⚠️  Error generating training curves: {e}")

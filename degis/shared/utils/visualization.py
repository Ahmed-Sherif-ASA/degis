"""
Visualization utilities for generated images and color palettes.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Optional, Dict, Any
from skimage.color import lab2rgb
import time


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
    size: int = 512,
    labels: Optional[List[str]] = None
) -> Image.Image:
    """Display multiple images in a grid layout with optional labels."""
    if not images:
        return None
    
    # Resize all images to the same size
    resized_images = [img.resize((size, size)) for img in images]
    
    w, h = size, size
    rows = (len(resized_images) + cols - 1) // cols
    
    # Add space for labels if provided
    label_height = 30 if labels else 0
    grid_w = cols * w + (cols - 1) * padding
    grid_h = rows * h + (rows - 1) * padding + rows * label_height
    
    grid_img = Image.new("RGB", (grid_w, grid_h), color=bg_color)
    
    # Add labels if provided
    if labels:
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(grid_img)
        
        # Try to load a font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None
    
    for i, img in enumerate(resized_images):
        row, col = divmod(i, cols)
        x = col * (w + padding)
        y = row * (h + padding) + row * label_height
        
        # Paste image
        grid_img.paste(img, (x, y + label_height))
        
        # Add label if provided
        if labels and i < len(labels):
            text = str(labels[i])
            if font:
                bbox = draw.textbbox((0, 0), text, font=font)
                text_w = bbox[2] - bbox[0]
            else:
                text_w = len(text) * 10
            
            text_x = x + (w - text_w) // 2
            text_y = y + 5
            
            # Draw label
            draw.rectangle([text_x-5, text_y-2, text_x+text_w+5, text_y+20], fill=(0, 0, 0))
            draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
    
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
    
    # Plot Sinkhorn losses
    if 'train_sinkhorn' in metrics and 'val_sinkhorn' in metrics:
        axes[0, 0].plot(metrics['train_sinkhorn'], label='Train Sinkhorn')
        axes[0, 0].plot(metrics['val_sinkhorn'], label='Val Sinkhorn')
        axes[0, 0].set_title('Sinkhorn Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Sinkhorn')
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
    Generate training curves (Sinkhorn and loss) from metrics CSV.
    
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
        
        # Sinkhorn curves
        plt.figure(figsize=(10, 6))
        plt.plot(df.epoch, df.train_sinkhorn, label="train", linewidth=2)
        plt.plot(df.epoch, df.val_sinkhorn, label="val", linewidth=2)
        plt.xlabel("epoch")
        plt.ylabel("Sinkhorn")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f"Sinkhorn Curves - {dataset_name}_{hist_kind}")
        plt.savefig(os.path.join(output_dir, "sinkhorn_curves.png"), dpi=150, bbox_inches='tight')
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
        
        print("✓ Generated training curves (sinkhorn_curves.png, loss_curve.png)")
        
    except ImportError:
        print("⚠️  matplotlib not available, skipping training curves")
    except Exception as e:
        print(f"⚠️  Error generating training curves: {e}")


def visualize_generation_comparison(
    color_source_image: Image.Image,
    edge_map_image: Image.Image,
    style_generated_image: Image.Image,
    sinkhorn_generated_image: Image.Image,
    color_histogram: np.ndarray,
    color_space: str = "lab",
    style_metrics: Optional[Dict[str, Any]] = None,
    sinkhorn_metrics: Optional[Dict[str, Any]] = None,
    grid_size: int = 256,
    font_size: int = 10,
    figsize: Tuple[int, int] = (12, 6)
) -> Image.Image:
    """
    Create a simple matplotlib-based visualization row showing generation comparison.
    Matches the clean style from the user's example.
    
    Row layout (left to right):
    1. Color source image + top 20 histogram bins
    2. Edge map image  
    3. Style generation result + metrics
    4. Sinkhorn generation result + metrics
    """
    import matplotlib.pyplot as plt
    from skimage.color import lab2rgb
    import torch
    
    # Helper function to get top palette colors (from user's example)
    def _top_palette(vec, bins=8, top_k=20, space=color_space, c_max=150.0):
        v = vec.detach().cpu().numpy() if isinstance(vec, torch.Tensor) else np.asarray(vec)
        core = v[:bins**3]  # ignore the last 2 BW slots for palette picking
        idxs = np.argsort(core)[-top_k:][::-1]
        cols = []
        if space == "rgb":
            for flat in idxs:
                ri = flat // (bins*bins); gi = (flat // bins) % bins; bi = flat % bins
                cols.append(((ri+0.5)/bins, (gi+0.5)/bins, (bi+0.5)/bins))
        elif space == "lab":
            for flat in idxs:
                Li = flat // (bins*bins); ai = (flat // bins) % bins; bi = flat % bins
                L = (Li+0.5)/bins*100.0
                a = (ai+0.5)/bins*255.0 - 128.0
                b = (bi+0.5)/bins*255.0 - 128.0
                cols.append(tuple(lab2rgb(np.array([[[L,a,b]]]))[0,0]))
        elif space == "hcl":
            for flat in idxs:
                Li = flat // (bins*bins); Ci = (flat // bins) % bins; Hi = flat % bins
                L = (Li+0.5)/bins*100.0
                C = (Ci+0.5)/bins*c_max
                H = (Hi+0.5)/bins*360.0
                a = C*np.cos(np.deg2rad(H)); b = C*np.sin(np.deg2rad(H))
                cols.append(tuple(lab2rgb(np.array([[[L,a,b]]]))[0,0]))
        else:
            raise ValueError("space must be 'rgb' | 'lab' | 'hcl'")
        return cols, core[idxs]
    
    # Helper function to plot palette
    def _plot_palette(ax, colors, values, title):
        for i, (c, v) in enumerate(zip(colors, values)):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
            # ax.text(i+0.5, -0.08, f"{v:.3f}", ha="center", va="top", fontsize=7)
        ax.set_xlim(0, len(colors)); ax.set_ylim(0,1); ax.axis("off")
        ax.set_title(title, fontsize=12)
    
    # Create figure with 2x4 layout
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    
    # Convert PIL images to numpy arrays
    color_np = np.array(color_source_image.resize((grid_size, grid_size)))
    edge_np = np.array(edge_map_image.resize((grid_size, grid_size)))
    style_np = np.array(style_generated_image.resize((grid_size, grid_size)))
    sinkhorn_np = np.array(sinkhorn_generated_image.resize((grid_size, grid_size)))
    
    # === FIRST ROW ===
    
    # 1. Style image (top-left)
    axes[0, 0].imshow(color_np)
    axes[0, 0].axis("off")
    axes[0, 0].set_title(f"Reference Image", fontsize=font_size)
    
    # 2. Edge map (top-center)
    axes[0, 1].imshow(edge_np, cmap='gray')
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Edge Map", fontsize=font_size)

    # 3. Generated by Whole Image Embedding (top-right)
    axes[0, 2].imshow(style_np)
    axes[0, 2].axis("off")
    axes[0, 2].set_title("Result", fontsize=font_size)

    # 4. Generation metrics (top-right)
    axes[0, 3].axis("off")
    style_text = "Base-line IP-Adapter Metrics:"
    if style_metrics:
        style_text += f"\nTime: {style_metrics.get('generation_time', 'N/A')}s"
        style_text += f"\nSinkhorn: {style_metrics.get('sinkhorn', 'N/A'):.4f}"
        style_text += f"\nCosine: {style_metrics.get('cosine', 'N/A'):.4f}"
        style_text += f"\nIoU: {style_metrics.get('iou', 'N/A'):.4f}"
    axes[0, 3].text(0.1, 0.9, style_text, transform=axes[0, 3].transAxes, 
                    fontsize=font_size, verticalalignment='top', fontweight='bold')
    
    # === SECOND ROW ===
    
    # 5. Histogram (bottom-left)
    hist_cols, hist_vals = _top_palette(color_histogram, top_k=14)
    _plot_palette(axes[1, 0], hist_cols, hist_vals, f"Reference Palette")
    
    # 6. Edge map (bottom-center) - same as top
    axes[1, 1].imshow(edge_np, cmap='gray')
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Edge Map", fontsize=font_size)
    
    # 7. Generated by Colour Embedding (bottom-right)
    axes[1, 2].imshow(sinkhorn_np)
    axes[1, 2].axis("off")
    axes[1, 2].set_title("Result", fontsize=font_size)
    
    # 8. Sinkhorn metrics (bottom-right)
    axes[1, 3].axis("off")
    sinkhorn_text = "DEGIS Metrics:"
    if sinkhorn_metrics:
        sinkhorn_text += f"\nTime: {sinkhorn_metrics.get('generation_time', 'N/A')}s"
        sinkhorn_text += f"\nSinkhorn: {sinkhorn_metrics.get('sinkhorn', 'N/A'):.4f}"
        sinkhorn_text += f"\nCosine: {sinkhorn_metrics.get('cosine', 'N/A'):.4f}"
        sinkhorn_text += f"\nIoU: {sinkhorn_metrics.get('iou', 'N/A'):.4f}"
        sinkhorn_text += f"\nAttempts: {sinkhorn_metrics.get('attempts', 'N/A')}"
    axes[1, 3].text(0.1, 0.9, sinkhorn_text, transform=axes[1, 3].transAxes, 
                    fontsize=font_size, verticalalignment='top', fontweight='bold')
    
    plt.tight_layout()
    
    # Convert matplotlib figure to PIL Image
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    pil_img = Image.open(buf)
    plt.close()
    
    return pil_img


def create_generation_metrics(
    generation_time: float,
    sinkhorn_score: float,
    cosine_score: float,
    attempts: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a metrics dictionary for generation visualization.
    
    Args:
        generation_time: Time taken for generation in seconds
        sinkhorn_score: Sinkhorn distance score
        cosine_score: Cosine similarity score
        attempts: Number of attempts (for Sinkhorn generation)
        
    Returns:
        Dictionary with formatted metrics
    """
    metrics = {
        'generation_time': f"{generation_time:.2f}",
        'sinkhorn': sinkhorn_score,
        'cosine': cosine_score
    }
    
    if attempts is not None:
        metrics['attempts'] = attempts
    
    return metrics

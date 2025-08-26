import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths
NAME = "laion_5m"
DATA_DIR = "/data/thesis/data"
MODELS_DIR = "/data/thesis/models"

CSV_PATH = f"/data/thesis/laion_5m_manifest.csv"
EMBEDDINGS_PATH = f"/data/thesis/data/laion_5m_clip_embeddings.npy"

COLOR_HIST_PATH_RGB = f"/data/thesis/data/laion_5m_color_histograms_rgb_512.npy"
COLOR_HIST_PATH_LAB = f"/data/thesis/data/laion_5m_color_histograms_lab_512.npy"
COLOR_HIST_PATH_HCL = f"/data/thesis/data/laion_5m_color_histograms_hcl_512.npy"
COLOR_HIST_PATH_LAB_514 = f"/data/thesis/data/laion_5m_color_histograms_lab_514.npy"
COLOR_HIST_PATH_HCL_514 = f"/data/thesis/data/laion_5m_color_histograms_hcl_514.npy"
EDGE_MAPS_PATH = f"/data/thesis/data/laion_5m_edge_maps.npy"
HF_HUB_CACHE = f"/data/thesis/data/models"
EMBEDDINGS_TARGET_PATH = f"/data/thesis/models/laion_5m_embeddings.npy"


BATCH_SIZE = 512
IMG_SIZE = (224,224)
import torch
import os

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache setup
MODEL_CACHE = os.getenv("DEGIS_CACHE_DIR", "./model-cache")
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE

# Dataset paths
# NAME = "laion_5m"
# NAME = "coco"
NAME = "test"
DATA_DIR = "/data/thesis/data"
MODELS_DIR = "/data/thesis/models"
EDGE_MAPS_PATH = f"/data/thesis/data/laion_5m_edge_maps.npy"
CSV_PATH = f"/data/thesis/laion_5m_manifest.csv"

EMBEDDINGS_PATH = f"/data/thesis/data/laion_5m_clip_embeddings.npy"
EMBEDDINGS_TARGET_PATH = f"/data/thesis/models/laion_5m_embeddings.npy"

HF_XL_EMBEDDINGS_TARGET_PATH = f"/data/thesis/models/hf_xl_laion_5m_embeddings.npy"



COLOR_HIST_PATH_RGB = f"/data/thesis/data/laion_5m_color_histograms_rgb_512.npy"
COLOR_HIST_PATH_LAB_514 = f"/data/thesis/data/laion_5m_color_histograms_lab_514.npy"
COLOR_HIST_PATH_HCL_514 = f"/data/thesis/data/laion_5m_color_histograms_hcl_514.npy"





BATCH_SIZE = 512
IMG_SIZE = (224,224)
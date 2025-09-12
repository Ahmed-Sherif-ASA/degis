import torch
import os

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model cache setup
MODEL_CACHE = os.getenv("DEGIS_CACHE_DIR", "./model-cache")
os.makedirs(MODEL_CACHE, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE

# Dataset paths
NAME = "laion_5m"
# NAME = "coco"
# NAME = "test"
DATA_DIR = "/data/thesis/data"
MODELS_DIR = "/data/thesis/models"
EDGE_MAPS_PATH = f"{DATA_DIR}/{NAME}_edge_maps.npy"
CSV_PATH = f"/data/thesis/{NAME}_manifest.csv"

EMBEDDINGS_PATH = f"{DATA_DIR}/{NAME}_clip_embeddings.npy"
EMBEDDINGS_TARGET_PATH = f"{MODELS_DIR}/{NAME}_embeddings.npy"

HF_XL_EMBEDDINGS_TARGET_PATH = f"{MODELS_DIR}/hf_xl_{NAME}_embeddings.npy"



COLOR_HIST_PATH_RGB = f"{DATA_DIR}/{NAME}_color_histograms_rgb_512.npy"
COLOR_HIST_PATH_LAB_514 = f"{DATA_DIR}/{NAME}_color_histograms_lab_514.npy"
COLOR_HIST_PATH_HCL_514 = f"{DATA_DIR}/{NAME}_color_histograms_hcl_514.npy"





BATCH_SIZE = 512
IMG_SIZE = (224,224)
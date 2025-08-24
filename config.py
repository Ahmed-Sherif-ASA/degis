import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths
NAME = "laion_5m"
DATA_DIR = "/data/thesis/data"
MODELS_DIR = "/data/thesis/models"

CSV_PATH = f"/data/thesis/{NAME}_manifest.csv"
EMBEDDINGS_PATH = f"{DATA_DIR}/{NAME}_clip_embeddings.npy"

COLOR_HIST_PATH_RGB = f"{DATA_DIR}/{NAME}_color_histograms_rgb_512.npy"
COLOR_HIST_PATH_LAB = f"{DATA_DIR}/{NAME}_color_histograms_lab_512.npy"
COLOR_HIST_PATH_HCL = f"{DATA_DIR}/{NAME}_color_histograms_hcl_512.npy"
COLOR_HIST_PATH_LAB_514 = f"{DATA_DIR}/{NAME}_color_histograms_lab_514.npy"
COLOR_HIST_PATH_HCL_514 = f"{DATA_DIR}/{NAME}_color_histograms_hcl_514.npy"
EDGE_MAPS_PATH = f"{DATA_DIR}/{NAME}_edge_maps.npy"
HF_HUB_CACHE = f"{DATA_DIR}/models"
EMBEDDINGS_TARGET_PATH = f"{MODELS_DIR}/{NAME}_embeddings.npy"


BATCH_SIZE = 512
IMG_SIZE = (224,224)
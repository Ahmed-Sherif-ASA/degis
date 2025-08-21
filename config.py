import torch

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset paths
NAME = "laion_1m"
DATA_DIR = "/data/thesis/data"

CSV_PATH = f"{DATA_DIR}/{NAME}_filtered_images.csv"
EMBEDDINGS_PATH = f"{DATA_DIR}/{NAME}_clip_embeddings.npy"

COLOR_HIST_PATH_RGB = f"{DATA_DIR}/{NAME}_color_histograms_rgb_512.npy"
COLOR_HIST_PATH_LAB = f"{DATA_DIR}/{NAME}_color_histograms_lab_512.npy"
COLOR_HIST_PATH_HCL = f"{DATA_DIR}/{NAME}_color_histograms_hcl_512.npy"

BATCH_SIZE = 512
IMG_SIZE = (224,224)
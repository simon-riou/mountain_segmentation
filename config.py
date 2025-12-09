import numpy as np
import os

# --- PATHS ---
DATA_DIR = "./data"
IMAGES_DIR = os.path.join(DATA_DIR, "images")
MASKS_DIR = os.path.join(DATA_DIR, "masks")
OUTPUT_DIR = "./segformer_montagne_output"
MODEL_CHECKPOINT = "nvidia/mit-b0"

# --- HYPERPARAMETERS ---
# --- HYPERPARAMETERS ---
LR = 0.00006
BATCH_SIZE = 4  # Ajuster selon VRAM (4 pour 8GB VRAM)
EPOCHS_FEATURE_EXTRACTION = 5
EPOCHS_FINE_TUNING = 45

# Learning Rates spécifiques
LR_FEATURE_EXTRACTION = 1e-3  # Plus élevé pour la tête (Decoder)
LR_FINE_TUNING = 5e-5         # Plus faible pour tout le modèle

# --- IMAGE PROCESSING ---
TARGET_W = 1024
TARGET_H = 512
INFERENCE_SIZE = (TARGET_W, TARGET_H)

# --- CLASSES & COLORS ---
# Format: [R, G, B]
CLASS_COLORS = np.array([
    [34, 139, 34],    # 0: Foret Sombre (Vert)
    [124, 252, 0],    # 4: Foret (Vert)
    [45, 189, 255],   # 1: Neige (Cyan/Bleu clair)
    [139, 69, 19],    # 2: Roche (Marron)
    [0, 0, 0]         # 3: Ciel (Noir)
])

ID2LABEL = {0: "Foret_sombre", 1: "Neige", 2: "Roche", 3: "Ciel", 4: "Foret"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# For Inference (consistent with training)
CLASSES = ID2LABEL
PALETTE = CLASS_COLORS

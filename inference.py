import torch
from torch import nn
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from PIL import Image
import numpy as np
import os
import glob
import re
from config import MODEL_CHECKPOINT, INFERENCE_SIZE, OUTPUT_DIR, INFERENCE_DIR
from utils import visualize_prediction

def get_latest_checkpoint(output_dir):
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    # 1. Try to find a valid checkpoint subfolder
    if checkpoints:
        # Sort by number
        checkpoints.sort(key=lambda x: int(re.search(r"checkpoint-(\d+)", x).group(1)))
        latest_checkpoint = checkpoints[-1]
        
        # Verify if it contains model files (e.g. config.json)
        if os.path.exists(os.path.join(latest_checkpoint, "config.json")):
            return latest_checkpoint
        else:
            print(f"⚠️ Checkpoint trouvé mais semble vide/invalide : {latest_checkpoint}")

    # 2. If no valid checkpoint found, check if the output_dir itself has the model
    if os.path.exists(os.path.join(output_dir, "config.json")):
        print(f"✅ Utilisation du modèle trouvé directement dans : {output_dir}")
        return output_dir

    return None

# Default model path (can be overridden)
DEFAULT_MODEL_PATH = get_latest_checkpoint(INFERENCE_DIR)

def load_model(model_path=None):
    if model_path is None:
        model_path = DEFAULT_MODEL_PATH
        if model_path is None:
             # Fallback if no checkpoint found, though this might fail if not trained
             print(f"⚠️ Aucun checkpoint trouvé dans {OUTPUT_DIR}. Utilisation de {MODEL_CHECKPOINT} (non entraîné).")
             model_path = MODEL_CHECKPOINT
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Chargement du modèle depuis {model_path} sur {device}...")
    
    model = AutoModelForSemanticSegmentation.from_pretrained(model_path).to(device)
    # Processor loads base config
    processor = AutoImageProcessor.from_pretrained(MODEL_CHECKPOINT)
    
    return model, processor, device

def predict_image(image_path, model, processor, device):
    # 1. Load and Resize Manually
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size # (W, H)
    
    input_image = original_image.resize(INFERENCE_SIZE, resample=Image.BILINEAR)
    
    # 2. Prepare (Processor)
    inputs = processor(
        images=input_image, 
        do_resize=False,
        do_normalize=True,
        return_tensors="pt"
    ).to(device)
    
    # 3. Inference
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 4. Post-Processing (Upsampling)
    logits = outputs.logits
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=input_image.size[::-1], # (H, W) expected
        mode="bilinear",
        align_corners=False
    )
    
    # 5. Argmax
    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    
    # 6. Resize mask to original size
    mask_image = Image.fromarray(pred_seg.astype(np.uint8))
    full_size_mask = mask_image.resize(original_size, resample=Image.NEAREST)
    
    return original_image, np.array(full_size_mask)

def main():
    # Example usage
    #test_image_path = "./data/new_images/Mont_Blanc_Aiguille_du_Midi_2016_03_11_12-00-00_full_panorama.jpg" # unseen
    test_image_path = "./data/images/Mont_Blanc_Aiguille_du_Midi_2019_01_29_10-30-00_full.jpg" # seen
    if os.path.exists(test_image_path):
        model, processor, device = load_model()
        orig_img, pred_mask = predict_image(test_image_path, model, processor, device)
        visualize_prediction(orig_img, pred_mask)
    else:
        print(f"Image de test non trouvée: {test_image_path}")
        print("Veuillez modifier le chemin dans inference.py ou fournir un chemin.")

if __name__ == "__main__":
    main()

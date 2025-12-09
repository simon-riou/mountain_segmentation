import os
import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from config import CLASS_COLORS, TARGET_W, TARGET_H
import cv2

def get_training_transforms():
    return A.Compose([
        # --- 1. Transformations Géométriques (Légères) ---
        # Flip horizontal OK (apprend la texture de la neige dans les deux sens)
        # PAS de VerticalFlip (le ciel n'est jamais en bas)
        A.HorizontalFlip(p=0.5),
        
        # Simulation de légers mouvements de caméra ou de vent
        A.Affine(
            translate_percent=(-0.1, 0.1), # Décalage max 10%
            scale=(0.85, 1.15),            # Zoom +/- 15%
            rotate=(-10, 10),              # Rotation +/- 10°
            border_mode=cv2.BORDER_REFLECT_101,   # Effet miroir sur les bords
            p=0.8
        ),
        
        # Déformation élastique légère pour varier la forme des crêtes
        # Empêche le modèle de mémoriser la silhouette exacte de la montagne
        A.ElasticTransform(
            alpha=1, 
            sigma=50, 
            # alpha_affine removed in 2.0
            p=0.2
        ),

        # --- 2. Transformations Colorimétriques (Agressives) ---
        # Crucial pour la neige qui change de couleur selon l'heure (bleu, orange, blanc)
        A.Compose([
            # Lumière / Contraste
            A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
            # Changement de teinte (Matin bleu vs Soir orange)
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.6),
        ], p=1.0),
        
        # Simulation d'ambiances spécifiques
        A.OneOf([
            A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0), # Changement de teinte global
            A.RandomGamma(p=1.0), # Gestion des ombres dures sur la roche
        ], p=0.5),

        # --- 3. Simulation Qualité / Météo ---
        # Simulation de brouillard ou de flou de mise au point
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        
        # Bruit numérique (ISO élevé le matin/soir)
        A.GaussNoise(p=0.3),

        # --- 4. Regularization (Anti-Overfitting critique pour petit dataset) ---
        # CoarseDropout : Crée des trous noirs dans l'image.
        # Force le modèle à utiliser le contexte spatial plutôt que de se focaliser sur une feature unique.
        A.CoarseDropout(
            num_holes_range=(2, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            p=0.5
        ),
    ])

class MountainDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor, transforms=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.transforms = transforms

        # 1. Find JPG images
        self.images = sorted(glob.glob(os.path.join(img_dir, "*_full.jpg")))

        if len(self.images) == 0:
             raise ValueError(f"Aucune image found dans {img_dir}")

        # 2. Match with PNG masks
        self.masks = []
        for img_path in self.images:
            filename = os.path.basename(img_path)

            # Replace extension with .png
            mask_name = filename.replace("_full.jpg", "_full_mask.png")
            mask_path = os.path.join(mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Masque introuvable: {mask_path}")

            self.masks.append(mask_path)

        print(f"✅ Dataset chargé : {len(self.images)} paires (Images JPG / Masques PNG)")

    def __len__(self):
        return len(self.images)

    def _rgb_to_mask(self, rgb_image):
        img_array = np.array(rgb_image)
        distances = np.linalg.norm(img_array[:, :, None, :] - CLASS_COLORS[None, None, :, :], axis=3)
        mask_indices = np.argmin(distances, axis=2)
        return mask_indices.astype(np.uint8)

    def __getitem__(self, idx):
        # 1. Open images
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("RGB")
        
        # 2. Resize (Always resize to target size first)
        image = image.resize((TARGET_W, TARGET_H), resample=Image.BILINEAR)
        # Use NEAREST for mask to avoid interpolating classes
        mask = mask.resize((TARGET_W, TARGET_H), resample=Image.NEAREST)
        
        # 3. Convert mask to indices (H, W)
        segmentation_map = self._rgb_to_mask(mask)
        
        # 4. Apply Augmentations (if any)
        if self.transforms:
            # Albumentations expects numpy arrays
            image_np = np.array(image)
            # segmentation_map is already numpy array from _rgb_to_mask
            
            transformed = self.transforms(image=image_np, mask=segmentation_map)
            image = Image.fromarray(transformed["image"])
            segmentation_map = Image.fromarray(transformed["mask"])
        else:
            segmentation_map = Image.fromarray(segmentation_map)

        # 5. Process
        inputs = self.processor(
            images=image, 
            segmentation_maps=segmentation_map, 
            return_tensors="pt",
            do_resize=False, # Already resized manually
            do_normalize=True
        )

        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": inputs.labels.squeeze()
        }

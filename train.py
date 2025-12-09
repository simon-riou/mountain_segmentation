import torch
from transformers import TrainingArguments, Trainer
from config import (
    MODEL_CHECKPOINT, IMAGES_DIR, MASKS_DIR, OUTPUT_DIR, 
    EPOCHS_FEATURE_EXTRACTION, EPOCHS_FINE_TUNING, BATCH_SIZE, ID2LABEL, LABEL2ID,
    LR_FEATURE_EXTRACTION, LR_FINE_TUNING
)
from dataset import MountainDataset, get_training_transforms
from model import get_model, get_processor, freeze_encoder
from utils import compute_metrics
import glob
import os

def main():
    # 1. Initialize Processor
    processor = get_processor(MODEL_CHECKPOINT)

    # 2. Split Dataset (Filenames)
    # We do this manually to apply transforms ONLY to train set
    all_images = sorted(glob.glob(os.path.join(IMAGES_DIR, "*_full.jpg")))
    
    # Simple split 90/10
    split_idx = int(0.9 * len(all_images))
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    print(f"Split: {len(train_images)} train, {len(val_images)} val")

    # 3. Create Datasets
    # Train dataset WITH transforms
    train_dataset = MountainDataset(
        img_dir=IMAGES_DIR,
        mask_dir=MASKS_DIR,
        processor=processor,
        transforms=get_training_transforms()
    )
    # Override internal list to only include train images
    # This is a bit hacky but avoids changing Dataset signature too much or creating a new class
    # Better approach: Pass list of files to Dataset, but Dataset currently scans dir.
    # Let's stick to the plan: We need to filter the dataset.
    # Actually, the Dataset scans the dir in __init__. 
    # To properly support this without changing Dataset signature to accept file list (which would be cleaner but requires more changes),
    # we can just filter `self.images` after init.
    
    train_dataset.images = train_images
    # Re-match masks for these images
    train_dataset.masks = [
        os.path.join(MASKS_DIR, os.path.basename(img).replace("_full.jpg", "_full_mask.png"))
        for img in train_images
    ]

    # Val dataset WITHOUT transforms
    val_dataset = MountainDataset(
        img_dir=IMAGES_DIR,
        mask_dir=MASKS_DIR,
        processor=processor,
        transforms=None
    )
    val_dataset.images = val_images
    val_dataset.masks = [
        os.path.join(MASKS_DIR, os.path.basename(img).replace("_full.jpg", "_full_mask.png"))
        for img in val_images
    ]

    # 4. Initialize Model
    model = get_model(MODEL_CHECKPOINT, ID2LABEL, LABEL2ID)

    # --- PHASE 1: FEATURE EXTRACTION ---
    print("\n=== PHASE 1: FEATURE EXTRACTION (Encoder Gelé) ===")
    freeze_encoder(model, freeze=True)
    
    args_phase1 = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "phase1_feature_extraction"),
        learning_rate=LR_FEATURE_EXTRACTION,
        num_train_epochs=EPOCHS_FEATURE_EXTRACTION,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUTPUT_DIR, "logs", "phase1"),
    )

    trainer_phase1 = Trainer(
        model=model,
        args=args_phase1,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer_phase1.train()
    
    # --- PHASE 2: FINE-TUNING ---
    print("\n=== PHASE 2: FINE-TUNING (Tout Dégelé) ===")
    freeze_encoder(model, freeze=False)
    
    args_phase2 = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "phase2_finetuning"),
        learning_rate=LR_FINE_TUNING,
        num_train_epochs=EPOCHS_FINE_TUNING,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUTPUT_DIR, "logs", "phase2"),
    )

    trainer_phase2 = Trainer(
        model=model,
        args=args_phase2,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer_phase2.train()
    
    # Save final model
    final_save_path = os.path.join(OUTPUT_DIR, "final_model")
    trainer_phase2.save_model(final_save_path)
    processor.save_pretrained(final_save_path)
    print(f"Modèle final sauvegardé dans {final_save_path}")

if __name__ == "__main__":
    main()

import torch
from transformers import TrainingArguments, Trainer
from config import (
    MODEL_CHECKPOINT, IMAGES_DIR, MASKS_DIR, OUTPUT_DIR, 
    LR, EPOCHS, BATCH_SIZE, ID2LABEL, LABEL2ID
)
from dataset import MountainDataset, get_training_transforms
from model import get_model, get_processor
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

    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # 7. Train
    print("Début de l'entraînement...")
    trainer.train()

if __name__ == "__main__":
    main()

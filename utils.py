import torch
import numpy as np
import matplotlib.pyplot as plt
import evaluate
from config import ID2LABEL, PALETTE

# Initialize metric
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # Resize logits to label size because SegFormer output is 1/4 size
    logits_tensor = torch.from_numpy(logits)
    logits_tensor = torch.nn.functional.interpolate(
        logits_tensor,
        size=labels.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).argmax(dim=1)

    pred_labels = logits_tensor.detach().cpu().numpy().astype(np.int32)
    labels = labels.astype(np.int32)

    metrics = metric.compute(
        predictions=pred_labels,
        references=labels,
        num_labels=len(ID2LABEL),
        ignore_index=255
    )

    return {
        "mean_iou": metrics["mean_iou"],
        "mean_accuracy": metrics["mean_accuracy"]
    }

def visualize_prediction(image, mask_array):
    # Colorize mask
    color_seg = np.zeros((mask_array.shape[0], mask_array.shape[1], 3), dtype=np.uint8)
    for label_id, color in enumerate(PALETTE):
        color_seg[mask_array == label_id] = color

    # Overlay
    img_np = np.array(image)
    overlay = (img_np * 0.6 + color_seg * 0.4).astype(np.uint8)

    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.imshow(img_np)
    plt.title("Image Originale")
    plt.axis('off')
    
    plt.subplot(3, 1, 2)
    plt.imshow(color_seg)
    plt.title("Masque de Segmentation")
    plt.axis('off')
    
    plt.subplot(3, 1, 3)
    plt.imshow(overlay)
    plt.title("Superposition")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from config import TARGET_H, TARGET_W

def get_processor(model_checkpoint):
    return SegformerImageProcessor.from_pretrained(
        model_checkpoint,
        do_resize=True,
        size={"height": TARGET_H, "width": TARGET_W},
        do_normalize=True,
    )

def get_model(model_checkpoint, id2label, label2id):
    return SegformerForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from config import TARGET_H, TARGET_W

def get_processor(model_checkpoint):
    return AutoImageProcessor.from_pretrained(
        model_checkpoint,
        do_resize=True,
        size={"height": TARGET_H, "width": TARGET_W},
        do_normalize=True,
    )

def get_model(model_checkpoint, id2label, label2id):
    return AutoModelForSemanticSegmentation.from_pretrained(
        model_checkpoint,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

def freeze_encoder(model, freeze=True):
    """
    Gèle ou dégèle l'encodeur (backbone) du modèle de manière générique.
    """
    # Récupère le modèle de base (backbone) via le préfixe
    # Attention: certains modèles (ex: UperNet) ont un prefix vide '' qui fait planter getattr
    if hasattr(model, "base_model_prefix") and model.base_model_prefix:
        base_model = getattr(model, model.base_model_prefix)
    else:
        base_model = model

    # Tente de trouver l'attribut 'encoder' ou 'backbone'
    if hasattr(base_model, "encoder"):
        encoder = base_model.encoder
    elif hasattr(base_model, "backbone"):
        encoder = base_model.backbone
    else:
        encoder = base_model
        
    for param in encoder.parameters():
        param.requires_grad = not freeze
    
    status = "gelé" if freeze else "dégelé"
    print(f"Encodeur ({type(encoder).__name__}) {status}.")

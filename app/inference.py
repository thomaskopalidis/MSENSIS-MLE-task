import io
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

FINETUNED_MODEL_PATH = Path("models/best_model")
PRETRAINED_MODEL_NAME = "google/vit-base-patch16-224"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


IMAGENET_CAT_INDICES = [281, 282, 283, 284, 285]  
IMAGENET_DOG_INDICES = list(range(151, 269))  

_models = {}  # cache: {"finetuned": (model, processor), "pretrained": (model, processor)}

"""
Inference module for the Cats & Dogs classifier.

Supports two models, selectable at call time:
- "finetuned": the ViT fine-tuned on the cats/dogs dataset (models/best_model/)
- "pretrained": the general-purpose ImageNet ViT (google/vit-base-patch16-224)
  with no fine-tuning.
"""
def load_model(which: str = "finetuned"):
    """
    Load and cache one of the two supported models.

    Args:
        which: "finetuned" for the cats/dogs fine-tuned ViT, or "pretrained"
            for the general ImageNet ViT baseline.

    Returns:
        tuple: (model, processor)
    """
    if which not in ("finetuned", "pretrained"):
        raise ValueError(f"Unknown model choice: {which}")

    if which not in _models:
        if which == "finetuned":
            model = ViTForImageClassification.from_pretrained(FINETUNED_MODEL_PATH).to(DEVICE)
            processor = ViTImageProcessor.from_pretrained(FINETUNED_MODEL_PATH)
        else:
            model = ViTForImageClassification.from_pretrained(PRETRAINED_MODEL_NAME).to(DEVICE)
            processor = ViTImageProcessor.from_pretrained(PRETRAINED_MODEL_NAME)
        model.eval()
        _models[which] = (model, processor)

    return _models[which]


def _predict_finetuned(image: Image.Image) -> dict:
    model, processor = load_model("finetuned")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = probs.argmax().item()

    return {
        "predicted_class": model.config.id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
        "model_used": "finetuned",
    }


def _predict_pretrained(image: Image.Image) -> dict:
    model, processor = load_model("pretrained")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    cat_prob = probs[IMAGENET_CAT_INDICES].sum().item()
    dog_prob = probs[IMAGENET_DOG_INDICES].sum().item()
    total = cat_prob + dog_prob

    if total == 0:
        # Neither a recognizable cat nor dog breed was in the top predictions.
        predicted_class = "unknown"
        confidence = 0.0
    else:
        predicted_class = "cat" if cat_prob > dog_prob else "dog"
        confidence = round(max(cat_prob, dog_prob) / total, 4)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "model_used": "pretrained",
    }


def predict_image(image_bytes: bytes, model_choice: str = "finetuned") -> dict:
    """
    Run inference on raw image bytes using the selected model.

    Args:
        image_bytes: Raw bytes of the uploaded image file.
        model_choice: "finetuned" or "pretrained".

    Returns:
        dict: {"predicted_class": str, "confidence": float, "model_used": str}
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if model_choice == "pretrained":
        return _predict_pretrained(image)
    return _predict_finetuned(image)
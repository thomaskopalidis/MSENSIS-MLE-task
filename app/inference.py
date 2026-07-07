"""
Inference module for the Cats & Dogs classifier.

Loads the trained ViT model once and exposes a single `predict_image`
function used both by the FastAPI service (api.py) and by any offline
scoring scripts.
"""

import io
from pathlib import Path

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_PATH = Path("models/best_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_processor = None


def load_model():
    """
    Load the fine-tuned ViT model and its image processor from disk.
    Cached at module level so repeated calls don't reload the weights.

    Returns:
        tuple: (model, processor)
    """
    global _model, _processor
    if _model is None or _processor is None:
        _model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(DEVICE)
        _processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        _model.eval()
    return _model, _processor


def predict_image(image_bytes: bytes) -> dict:
    """
    Run inference on raw image bytes and return the predicted class and confidence.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        dict: {"predicted_class": str, "confidence": float}
    """
    model, processor = load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = probs.argmax().item()

    return {
        "predicted_class": model.config.id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
    }
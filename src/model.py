from transformers import ViTImageProcessor, ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"
ID2LABEL = {0: "cat", 1: "dog"}
LABEL2ID = {"cat": 0, "dog": 1}

def build_model():
    """
    Build a Vision Transformer (ViT) model for image classification.
    Load the pre-trained ViT model and replace its classification head for 
    binary cat/dog classification.
     
    Returns:
        tuple: (model, processor) where model is the ViT model and processor is the image processor.
    """

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,  # Ignore size mismatch when loading pre-trained weights
    )
    return model, processor
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Literal

from app.inference import predict_image, load_model

app = FastAPI(title="Cats & Dogs Classifier API")

# Pre-load both models at startup rather than on first request per model.
load_model("finetuned")
load_model("pretrained")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_choice: Literal["finetuned", "pretrained"] = Form("finetuned"),
):
    """
    Classify an uploaded image as cat or dog.

    - model_choice="finetuned": uses the ViT fine-tuned on the cats/dogs dataset.
    - model_choice="pretrained": uses the general-purpose ImageNet ViT with no
      fine-tuning, mapping its predictions down to cat/dog.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()

    try:
        result = predict_image(contents, model_choice=model_choice)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    return result
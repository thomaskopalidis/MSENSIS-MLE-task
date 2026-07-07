from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch
import io
from transformers import ViTForImageClassification, ViTImageProcessor

app = FastAPI(title="Cats & Dogs Classifier API")

MODEL_PATH = "models/best_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViTForImageClassification.from_pretrained(MODEL_PATH).to(device)
processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
model.eval()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_id = probs.argmax().item()

    return {
        "predicted_class": model.config.id2label[pred_id],
        "confidence": round(probs[pred_id].item(), 4),
    }
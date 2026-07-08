# Cats & Dogs Classifier

A fine-tuned Vision Transformer (ViT) that classifies images as cat or dog,
served through a FastAPI backend with a Streamlit front-end.

## Project structure

```

├── app/
│   ├── api.py              # FastAPI service (/health, /predict)
│   ├── inference.py         # Model loading + prediction logic (both models)
│   └── streamlit_app.py     # Streamlit UI, calls the API
├── src/
│   ├── data_preprocessing.py  # Builds manifest, stratified train/val/test split
│   ├── cd_dataset.py           # PyTorch Dataset for cats/dogs
│   ├── model.py                # ViT model + processor builder
│   └── train.py                # Training loop + final test-set │   └── evaluate.py                 # Standalone test-set evaluation + confusion matrix plot
├── notebooks/
│   └── EDA.ipynb                # Data exploration (class balance, image sizes)
├── models/
│   └── best_model/               # Saved checkpoint (config, weights, preprocessor)
├── data/
│   ├── raw/                       # Raw images (cats/, dogs/) - not tracked in git
│   └── processed/                 # train.csv, val.csv, test.csv
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data preparation

Place raw images under `data/raw/cats/` and `data/raw/dogs/`, then run:

```bash
python src/data_preprocessing.py
```

This creates `data/processed/train.csv`, `val.csv`, and `test.csv` (80/10/10
stratified split).

## Training

```bash
python src/train.py
```

Fine-tunes a ViT (`google/vit-base-patch16-224`) for binary
classification. The best checkpoint (by validation accuracy) is saved to
`models/best_model/`.


## Evaluation 
```bash
python src/evaluate.py
```

Loads the saved checkpoint and reports accuracy, precision, recall, and F1 on the held-out test, prints a metrics table to the console, and saves a confusion matrix plot to `confusion_matrix.png`
## Running the app

Start the API (from the project root):

```bash
uvicorn app.api:app --reload --port 8000
```

Check it's healthy:

```bash
curl http://localhost:8000/health
```

Interactive API docs: http://localhost:8000/docs

`/predict` accepts an image file plus a `model_choice` field (`"finetuned"`
or `"pretrained"`) so the caller can pick which model runs inference.


In a second terminal, start the UI:

```bash
streamlit run app/streamlit_app.py
```

Upload an image of a cat or dog and click "Classify" to get a prediction.

## Notes

- `app/__init__.py` must exist (can be empty) for `uvicorn app.api:app` and
  the `from app.inference import ...` import in `api.py` to work.
- The model expects a `models/best_model/` directory in HuggingFace format
  (`config.json`, `model.safetensors`, `preprocessor_config.json`), produced
  automatically by `train.py`. The pretrained baseline (google/vit-base-patch16-224) is downloaded automatically from the Hugging Face Gub on first run. 

## Copyright

© 2026 Thomas Kopalidis. All rights reserved. This repository is shared
publicly for review purposes only; no license is granted for reuse,
redistribution, or modification without explicit permission.

"""
Basic tests for the Cats & Dogs Classifier API.

Run from the project root with:
    pytest tests/test_api.py -v

Requires a valid models/best_model/ checkpoint on disk (the tests load the
real model, they are not mocked).
"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.api import app

client = TestClient(app)


def _make_dummy_image_bytes() -> bytes:
    """Create a small in-memory RGB image for testing, no external file needed."""
    img = Image.new("RGB", (224, 224), color=(120, 80, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_valid_image():
    image_bytes = _make_dummy_image_bytes()
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] in ("cat", "dog")
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_rejects_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400
import streamlit as st
import requests
from PIL import Image

st.title("Cats & Dogs Classifier")
st.write("Upload an image of a cat or dog to classify it.")

API_URL = "http://localhost:8000/predict"

model_choice = st.selectbox(
    "Select model",
    options=["finetuned", "pretrained"],
    format_func=lambda x: "Fine-tuned (trained on the cats/dogs dataset)" if x == "finetuned"
    else "Pretrained ViT (ImageNet, no fine-tuning)",
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Classify"):
        with st.spinner("Running prediction..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    data={"model_choice": model_choice},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Make sure it's running at http://localhost:8000.")
            else:
                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"Prediction: **{result['predicted_class']}** "
                        f"({result['confidence']*100:.1f}% confidence) "
                        f"— model used: {result['model_used']}"
                    )
                else:
                    st.error(f"Something went wrong on the API side (status {response.status_code}).")
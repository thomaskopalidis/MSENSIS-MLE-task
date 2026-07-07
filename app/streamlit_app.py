import streamlit as st
import requests
from PIL import Image

st.title("Cats & Dogs Classifier")
st.write("Ανεβάστε μια εικόνα γάτας ή σκύλου για ταξινόμηση.")

API_URL = "http://localhost:8000/predict"

uploaded_file = st.file_uploader("Επιλέξτε εικόνα", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Ταξινόμηση"):
        with st.spinner("Πρόβλεψη σε εξέλιξη..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Δεν ήταν δυνατή η σύνδεση με το API. Βεβαιωθείτε ότι τρέχει στο http://localhost:8000.")
            else:
                if response.status_code == 200:
                    result = response.json()
                    st.success(
                        f"Πρόβλεψη: **{result['predicted_class']}** "
                        f"({result['confidence']*100:.1f}% confidence)"
                    )
                else:
                    st.error(f"Κάτι πήγε στραβά στο API (status {response.status_code}).")
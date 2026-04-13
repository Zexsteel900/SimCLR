import streamlit as st
from inference import predict
from PIL import Image

st.title("Satellite Image Classification (SimCLR)")

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg", "webp"])

if uploaded:
    # Save image
    with open("temp.jpg", "wb") as f:
        f.write(uploaded.read())

    # Display image
    image = Image.open("temp.jpg")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run prediction
    label, confidence = predict("temp.jpg")

    # Display result
    st.subheader("Prediction")
    st.write(f"Class: **{label}**")
    st.write(f"Confidence: **{confidence:.2f}**")
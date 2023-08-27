import streamlit as st
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import Image

model = ResNet50(weights='imagenet')

st.title('Bilderkennung mit ResNet')

uploaded_image = st.file_uploader("Bild auswählen", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Hochgeladenes Bild', use_column_width=True)

    # Konvertiere PIL Image in NumPy Array
    image_np = np.array(image)

    # Vorverarbeitung des Bildes
    processed_image = np.array(image.resize((224, 224)))
    processed_image = preprocess_input(processed_image)
    processed_image = np.expand_dims(processed_image, axis=0)

    # Vorhersagen mit dem Modell
    predictions = model.predict(processed_image)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    st.subheader('Vorhersagen:')
    for _, label, score in decoded_predictions:
        st.write(f"{label}: {score:.2f}")

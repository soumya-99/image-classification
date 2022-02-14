import streamlit as st
import cv2
import numpy as np
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.models import load_model

model = load_model("models/mdl_wts.hdf5")
uploaded_file = st.file_uploader("Choose an Image", type="jpg")

map_dict = {0: 'dog',
            1: 'horse',
            2: 'elephant',
            3: 'butterfly',
            4: 'chicken',
            5: 'cat',
            6: 'cow',
            7: 'sheep',
            8: 'spider',
            9: 'squirrel'}

if uploaded_file is not None:
    # Converting file into OpenCV image
    file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (224, 224))

    # Show the image in page
    st.image(img, channels="RGB")
    resized = preprocess_input(resized)
    img_reshape = resized[np.newaxis, ...]

    # Predicting the class
    genarate_prediction = st.button("Generate Prediction")
    if genarate_prediction:
        prediction = model.predict(img_reshape).argmax()
        st.title(f"Prediction: This is a {map_dict[prediction]}.")

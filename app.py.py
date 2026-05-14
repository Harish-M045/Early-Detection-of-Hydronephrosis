
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("models/hydronephrosis_model.h5")

class_names = ["Normal", "Hydronephrosis"]

st.title("Hydronephrosis Detection using Ultrasound Images")

st.write("Upload an ultrasound image to classify whether it is Normal or Hydronephrosis.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.subheader("Prediction Result")

    if predicted_class == 0:
        st.success("Normal Kidney")
    else:
        st.error("Hydronephrosis Detected")

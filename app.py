import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


st.set_page_config(
    page_title="Cancer Cell Detection",
    page_icon="🧬",
    layout="wide"
)


MODEL_PATH = os.path.join("models", "best_model.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()



st.sidebar.title(" Cancer Cell Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Prediction",
        "Model Information",
        "About Project"
    ]
)


if page == "Prediction":

    st.title("Cancer Cell Detection using EfficientNetB0")

    st.markdown(
      
    )

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Uploaded Image")

            st.image(
                image,
                use_container_width=True
            )

        with col2:

            if st.button("Predict"):

                img = image.resize((224,224))

                img = img_to_array(img)

                img = img / 255.0

                img = np.expand_dims(img, axis=0)

                prediction = model.predict(img, verbose=0)[0][0]

                if prediction >= 0.5:

                    label = "Malignant"

                    confidence = prediction

                    color = "red"

                else:

                    label = "Benign"

                    confidence = 1 - prediction

                    color = "green"

                st.subheader("Prediction")

                if label == "Benign":

                    st.success(f" {label}")

                else:

                    st.error(f" {label}")

                st.subheader("Confidence")

                st.progress(float(confidence))

                st.metric(
                    "Confidence",
                    f"{confidence*100:.2f}%"
                )

                st.divider()

                st.subheader("Model")

                st.info("EfficientNetB0")

                st.subheader("Input Size")

                st.info("224 × 224 RGB")



elif page == "Model Information":

    st.title(" Model Information")

    st.markdown("Architecture")

    st.write("EfficientNetB0")

    st.markdown(" Task")

    st.write("Binary Image Classification")

    st.markdown("Classes")

    st.write("0 → Benign")
    st.write("1 → Malignant")

    st.markdown("Input Size")

    st.write("224 × 224 RGB")

    st.markdown("Loss Function")

    st.code("Binary Crossentropy")

    st.markdown("Optimizer")

    st.code("Adam")


else:

    st.title(" About Project")

    st.markdown(
         )
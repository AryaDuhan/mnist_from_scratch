import streamlit as st
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import os

# import the nns
from script import NeuralNetwork as SigmoidNN
from script_relu import NeuralNetwork as ReLUNN

# page title and layout
st.set_page_config(page_title="MNIST Model Comparison", layout="wide")
st.title("🧠 Sigmoid vs. ReLU/Softmax Neural Network Comparison")

# load model
@st.cache_resource
def load_sigmoid_model():
    try:
        with open('trained_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: trained_model.pkl not found for the Sigmoid model.")
        st.stop()

@st.cache_resource
def load_relu_model():
    try:
        with open('trained_model_relu.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: trained_model_relu.pkl not found for the ReLU/Softmax model.")
        st.stop()

# image processing
def preprocess_image(img_array):
    pil_image = Image.fromarray(img_array.astype('uint8')).convert('L')
    
    bbox = pil_image.getbbox()
    if bbox is None:
        return np.zeros((784, 1))

    cropped_image = pil_image.crop(bbox)
    width, height = cropped_image.size
    padding = abs(width - height) // 2
    padding_tuple = (padding, 0, padding, 0) if width > height else (0, padding, 0, padding)
    
    padded_image = ImageOps.expand(cropped_image, padding_tuple, fill=0)
    resized_image = padded_image.resize((20, 20), Image.Resampling.LANCZOS)
    
    final_image = Image.new('L', (28, 28), 0)
    final_image.paste(resized_image, (4, 4))
    
    img_final_array = np.array(final_image)
    return img_final_array.astype(np.float32).flatten().reshape(784, 1) / 255.0

# load models
sigmoid_nn = load_sigmoid_model()
relu_nn = load_relu_model()

# main app
col1, col2 = st.columns(2)

# sigmoid model selections
with col1:
    st.header("🔹 Model 1: Sigmoid Network")
    st.write("A simple network with one hidden layer and Sigmoid activation.")
    
    # canvas for sigmoid
    sigmoid_canvas = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_sigmoid",
    )
    
    sigmoid_predict_button = st.button("Predict with Sigmoid Model", use_container_width=True, type="primary")

    if sigmoid_predict_button:
        if sigmoid_canvas.image_data is not None and sigmoid_canvas.image_data.any():
            with st.spinner("Sigmoid model is thinking..."):
                processed_input = preprocess_image(sigmoid_canvas.image_data)
                
                # prediction for sigmoid
                _, final_output = sigmoid_nn.feedforward(processed_input)
                prediction = np.argmax(final_output)
                confidence = np.max(final_output) * 100
                
            st.markdown(f"#### Predicted Digit: **{prediction}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.bar_chart(final_output.flatten())
        else:
            st.info("Please draw a digit on the Sigmoid canvas first.")

# relu model
with col2:
    st.header("🔸 Model 2: ReLU/Softmax Network")
    st.write("A deeper network with two hidden layers, ReLU, and Softmax.")

    # canvas for relu
    relu_canvas = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas_relu",
    )

    relu_predict_button = st.button("Predict with ReLU/Softmax Model", use_container_width=True, type="primary")

    if relu_predict_button:
        if relu_canvas.image_data is not None and relu_canvas.image_data.any():
            with st.spinner("ReLU/Softmax model is thinking..."):
                processed_input = preprocess_image(relu_canvas.image_data)

                # prediction for relu
                *_, final_output = relu_nn.feedforward(processed_input)
                prediction = np.argmax(final_output)
                confidence = np.max(final_output) * 100

            st.markdown(f"#### Predicted Digit: **{prediction}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            st.bar_chart(final_output.flatten())
        else:
            st.info("Please draw a digit on the ReLU/Softmax canvas first.")
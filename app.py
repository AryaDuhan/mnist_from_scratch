import streamlit as st
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# page tittle and layout
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

# load model only once
@st.cache_resource
def load_model():
    try:
        with open('trained_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: trained_model.pkl not found. Please run the original script.py to train and save the model.")
        st.stop()

# load the model
nn = load_model()

# side bar
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This app recognizes handwritten digits drawn on a canvas using a simple Neural Network trained from scratch. It showcases the core concepts of feedforward and backpropagation.
    """)
    st.write("---")
    st.header("Tech Stack")
    st.markdown("""
    - **Model:** From scratch with NumPy
    - **Framework:** Streamlit
    - **Libraries:** Streamlit, Streamlit-Drawable-Canvas, NumPy, Pillow
    """)

# main app
st.title("MNIST Digit Recognizer")
st.write("Draw a single digit in the canvas below and click 'Predict'.")

# canvas

canvas_key = "canvas"

if "clear_canvas" not in st.session_state:
    st.session_state.clear_canvas = False

col1, col2 = st.columns([1, 1])

with col1:
    predict_button = st.button("Predict")
with col2:
    if st.button("Clear"):
        st.session_state.clear_canvas = True
        st.experimental_rerun()

canvas_result = st_canvas(
    stroke_width=20,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=canvas_key if not st.session_state.clear_canvas else "new_canvas",
)

if st.session_state.clear_canvas:
    st.session_state.canvas_canvas = False

# button to predict
if predict_button:
    if canvas_result.image_data is not None:
        with st.spinner("Predicting..."):
            img_array = canvas_result.image_data

            # convert to image and resize
            pil_image = Image.fromarray(img_array.astype('uint8')).convert('L')
            resized_image = pil_image.resize((28, 28), Image.Resampling.LANCZOS)

            # image to input 
            processed_input = np.array(resized_image).flatten().reshape(784, 1) / 255.0

            # make a prediction
            _, final_output = nn.feedforward(processed_input)
            prediction = np.argmax(final_output)

        st.success(f"Prediction: {prediction}")
    else:
        st.info("Please draw a digit in the canvas.")
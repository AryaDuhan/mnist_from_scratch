import streamlit as st
import numpy as np
import pickle
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from script import NeuralNetwork, load_idx_data # Make sure to import load_idx_data
import cv2
import os # Import os for file path operations

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

@st.cache_data
def get_test_accuracy(_nn):
    """Calculate and cache the model's accuracy on the MNIST test set."""
    path = "mnist_data"
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')
    
    if not os.path.exists(test_images_path) or not os.path.exists(test_labels_path):
        st.sidebar.error("MNIST test data not found. Cannot calculate accuracy.")
        return None
        
    X_test, Y_test = load_idx_data(test_images_path, test_labels_path)
    X_test = X_test / 255.0
    
    correct_predictions = 0
    for i in range(len(Y_test)):
        inputs = X_test[i].reshape(784, 1)
        label = Y_test[i]
        prediction = _nn.predict(inputs)
        if prediction == label:
            correct_predictions += 1
            
    return (correct_predictions / len(Y_test)) * 100

def preprocess_image(img_array):
    """ Preprocess the image for prediction. """
    
    # convert to image and resize
    pil_image = Image.fromarray(img_array.astype('uint8')).convert('L')
    
    # bounding box
    bbox = pil_image.getbbox()
    if bbox is None:
        return np.zeros((784, 1))

    # crop image to bbox
    cropped_image = pil_image.crop(bbox)

    # padding to make img square
    width, height = cropped_image.size
    padding = abs(width - height) // 2
    if width > height:
        padding_tuple = (0, padding, 0, padding)
    else:
        padding_tuple = (padding, 0, padding, 0)

    padded_image = ImageOps.expand(cropped_image, padding_tuple, fill=0)

    # resize
    resized_image = padded_image.resize((20, 20), Image.Resampling.LANCZOS)

    # new blac 28x28 img to paste the digit in center
    final_image = Image.new('L', (28, 28), 0)
    final_image.paste(resized_image, (4, 4))

    # convert to numpy and normalise 
    img_final_array = np.array(final_image)
    processed_input = img_final_array.astype(np.float32).flatten().reshape(784, 1) / 255.0

    return processed_input

# load the model
nn = load_model()

# side bar
with st.sidebar:
    st.header("About This Project")
    st.write("""
    This app recognizes handwritten digits drawn on a canvas using a simple Neural Network trained from scratch. It showcases the core concepts of feedforward and backpropagation.
    """)
    st.write("---")
    
    st.header("Model Performance")
    accuracy = get_test_accuracy(nn)
    if accuracy:
        st.metric(label="Test Accuracy", value=f"{accuracy:.2f}%")
    
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
col1, col2 = st.columns([1.5, 2])
with col1:
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    predict_button = st.button("Predict", use_container_width=True, type="primary")


# button to predict
with col2:
    st.subheader("Prediction")
    if predict_button:
        if canvas_result.image_data is not None and canvas_result.image_data.any():
            with st.spinner("Predicting..."):
                img_array = canvas_result.image_data

                # *** THIS IS THE FIX: Call your preprocess_image function ***
                processed_input = preprocess_image(img_array)

                # make a prediction
                _, final_output = nn.feedforward(processed_input)
                prediction = np.argmax(final_output)
                confidence = np.max(final_output) * 100
            
            # Display prediction and confidence
            st.markdown(f"## Predicted Digit: **{prediction}**")
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
            
            # Display probability chart
            st.write("Probabilities for each digit:")
            st.bar_chart(final_output.flatten())

        else:
            st.info("Please draw a digit in the canvas.")
    else:
        st.write("Draw a digit and click the predict button to see the results here.")
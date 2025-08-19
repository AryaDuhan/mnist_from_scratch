# 🧠 MNIST From Scratch – Pure Python, NumPy & Streamlit

A foundational **machine learning project** demonstrating a deep understanding of **neural network mechanics**.  
This code builds a **digit recognizer** entirely **from scratch** using only Python and NumPy—no TensorFlow, PyTorch, or Keras.

It also includes a **Streamlit web app** where you can **draw digits** and get predictions in real time.

---

## 🎨 Example – Streamlit App

#### ✅ Model Prediction

<table align="center">
  <tr>
    <td>
      <img src="screenshots/Screenshot 2025-08-20 005325.png"  width="280px">
    </td>
    <td>
      <img src="screenshots/Screenshot 2025-08-20 005423.png"  width="280px">
    </td>
  </tr>
</table>
---

## ✨ Key Features

- **Pure Python + NumPy** – Built entirely from scratch to showcase the underlying math.
- **Object-Oriented Design** – Neural network logic encapsulated in a clean `NeuralNetwork` class.
- **Backpropagation Algorithm** – Implemented manually without ML frameworks.
- **Smart Initialization** – Uses **Xavier/Glorot initialization** for stable and efficient training.
- **Data Preprocessing** – Normalization & shuffling for optimal learning.
- **Interactive App** – Draw digits on a canvas and predict with the trained model.
- **No Heavy Frameworks** – Only `numpy`, `streamlit`, `PIL` are required.

---

## 🏗 How It Works

### **Neural Network Architecture**

| Layer        | Nodes | Activation |
| :----------- | :---- | :--------- |
| Input        | 784   | —          |
| Hidden Layer | 100   | Sigmoid    |
| Output       | 10    | Sigmoid    |

### **Forward Propagation**

1. Input image (28×28 pixels) is **flattened** into a 784-element vector.
2. Multiply by **weights** and add **bias**.
3. Pass the result through the **activation function**.
4. Repeat until final predictions are produced.

### **Backpropagation**

1. Compute error using **Mean Squared Error (MSE)** loss.
2. Propagate error **backward** from output → hidden.
3. Calculate **gradients** w.r.t. weights and biases.
4. Update parameters using **Gradient Descent**.

---

## 🚀 Streamlit Web App

After training, you can interact with the model through a **Streamlit app**.

- Draw a digit in the **canvas**.
- Press **Predict** to classify it.
- The app preprocesses the image to match MNIST format before feeding it to the model.

---

## 📦 Installation & Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/AryaDuhan/mnist_from_scratch.git
cd mnist_from_scratch
```

### **2. Install Dependencies**

```bash
pip install numpy pillow streamlit streamlit-drawable-canvas
```

### **3. Download MNIST Dataset**

Download the `.idx` files from:  
🔗 [Kaggle – MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

Place them inside the `mnist_data/` folder:

```
mnist_data/
│── train-images.idx3-ubyte
│── train-labels.idx1-ubyte
│── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### **4. Train the Model**

```bash
python script.py
```

This will train the neural network and save the weights as `trained_model.pkl`.

### **5. Run the Streamlit App**

```bash
streamlit run app.py
```

This opens a local web app where you can **draw digits** and test predictions live.

---

## 📊 Example Output (Training)

```bash
Starting training on 60000 images for 3 epochs.
Epoch 1/3
Epoch 2/3
Epoch 3/3
Time: 24.84 seconds
Accuracy: 96.31%
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

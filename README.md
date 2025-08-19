# 🧠 MNIST From Scratch – Sigmoid vs ReLU/Softmax  

A foundational **machine learning project** demonstrating a deep understanding of **neural network mechanics**.  
This code builds **digit recognizers from scratch** using only Python and NumPy — no TensorFlow, PyTorch, or Keras.  

It includes **two neural network implementations**:  
- A **simple Sigmoid network** (educational baseline).  
- A **modern ReLU + Softmax network** (higher accuracy).  

Both models are integrated into a **Streamlit app** where you can **draw digits** and see predictions in real time.  

---

## 🎨 Example – Streamlit App  

### ✅ Model Predictions  

<table align="center">
<tr>
<td align="center"><b>Sigmoid Model – Drawing</b><br><img src="screenshots/sigmoid_draw.png" width="280px"></td>
<td align="center"><b>Sigmoid Model – Prediction</b><br><img src="screenshots/sigmoid_pred.png" width="280px"></td>
</tr>
<tr>
<td align="center"><b>ReLU/Softmax Model – Drawing</b><br><img src="screenshots/relu_draw.png" width="280px"></td>
<td align="center"><b>ReLU/Softmax Model – Prediction</b><br><img src="screenshots/relu_pred.png" width="280px"></td>
</tr>
</table>

---

## ✨ Key Features  

- **From Scratch Implementation** – No ML frameworks, just Python + NumPy.  
- **Two Architectures** – Compare a simple Sigmoid NN vs. modern ReLU/Softmax NN.  
- **Object-Oriented Design** – Neural network logic encapsulated in clean classes.  
- **Manual Backpropagation** – Implemented step by step for full transparency.  
- **Smart Initialization** – Xavier/Glorot for stable training.  
- **Interactive App** – Draw digits and get predictions live.  

---

## 🏗 Architectural Comparison  

This project provides a **side-by-side comparison** of a classic and a modern NN architecture, both built **from scratch**.  

### 🔹 Model 1: Simple Sigmoid Network (`script.py`)  

**Architecture**:  
- Input Layer (784 nodes)  
- One Hidden Layer (100 nodes)  
- Output Layer (10 nodes)  
- **Sigmoid** activations  

✅ **Pros**: Easy to understand fundamentals of feedforward/backprop.  
❌ **Cons**: Vanishing gradients, slower learning, ~95% accuracy.  

---

### 🔹 Model 2: ReLU/Softmax Network (`script_relu.py`)  

**Architecture**:  
- Input Layer (784 nodes)  
- Hidden Layer 1 (128 nodes, ReLU)  
- Hidden Layer 2 (64 nodes, ReLU)  
- Output Layer (10 nodes, Softmax)  

✅ **Pros**: Faster training, avoids vanishing gradient, ~97.5% accuracy.  
✅ **Probabilistic output** with confidence scores.  

---

### 📊 Performance at a Glance  

| Feature             | Sigmoid Model | ReLU/Softmax Model |
|---------------------|--------------|--------------------|
| Hidden Layers       | 1            | 2                  |
| Hidden Activations  | Sigmoid      | ReLU               |
| Output Activation   | Sigmoid      | Softmax            |
| Typical Accuracy    | ~95%         | ~97.5%             |
| Training Speed      | Slower       | Faster             |

---

## 📦 Installation & Setup  

### 1. Clone the Repository  

```bash
git clone https://github.com/AryaDuhan/mnist_from_scratch.git
cd mnist_from_scratch
```

### 2. Install Dependencies

```bash
pip install numpy pillow streamlit streamlit-drawable-canvas
```

### 3. Download MNIST Dataset

Download from: 🔗 [Kaggle – MNIST Dataset](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)

Place files in `mnist_data/`:

```
mnist_data/
│── train-images.idx3-ubyte
│── train-labels.idx1-ubyte
│── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

### 4. Train the Models

Train the **Sigmoid model**:

```bash
python script.py
```

Creates: `trained_model.pkl`

Train the **ReLU/Softmax model**:

```bash
python script_relu.py
```

Creates: `trained_model_relu.pkl`

### 5. Run the Streamlit App

Sigmoid model:

```bash
streamlit run app.py
```

ReLU/Softmax model:

```bash
streamlit run app_relu.py
```

---

## 📊 Example Training Output

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

This project is licensed under the **MIT License** – see the LICENSE file for details.
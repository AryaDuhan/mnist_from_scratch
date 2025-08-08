# 🧠 MNIST From Scratch – Pure Python & NumPy

A foundational **machine learning project** demonstrating a deep understanding of **neural network mechanics**. This code builds a **digit recognizer** entirely **from scratch** using only Python and NumPy—no TensorFlow, PyTorch, or Keras.

It implements the **core algorithms** of **feedforward** and **backpropagation** to train a model that classifies handwritten digits from the **MNIST dataset**.

---

## ✨ Key Features

-   **Pure Python + NumPy** – Built entirely from scratch to showcase the underlying math.
-   **Object-Oriented Design** – Neural network logic encapsulated in a clean `NeuralNetwork` class.
-   **Backpropagation Algorithm** – Implemented manually without ML frameworks.
-   **Smart Initialization** – Uses **Xavier/Glorot initialization** for stable and efficient training.
-   **Data Preprocessing** – Normalization & shuffling for optimal learning.
-   **No Framework Dependencies** – Only `numpy` is required for the core logic.

---

## 🏗 How It Works

This project uses a **fully connected neural network** with **one hidden layer**.

### **Architecture**

| Layer        | Nodes | Activation |
| :----------- | :---- | :--------- |
| Input        | 784   | —          |
| Hidden Layer | 100   | Sigmoid    |
| Output       | 10    | Sigmoid    |



### **Forward Propagation**

1.  Input image (28×28 pixels) is **flattened** into a 784-element vector.
2.  Multiply by **weights** and add **bias**.
3.  Pass the result through the **activation function** to get outputs for the next layer.
4.  Repeat for each layer until final predictions are produced.

### **Backpropagation**

1.  Compute error using **Mean Squared Error (MSE)** loss.
2.  Propagate error **backward** from the output layer to the hidden layer.
3.  Calculate **gradients** of the loss w.r.t. weights and biases.
4.  Update parameters using **Gradient Descent**.

### **Training Loop**

-   Run for several **epochs** over the training set.
-   **Shuffle data** each epoch to improve generalization.
-   Gradually **reduce error** until high accuracy is achieved.

---

## 📦 Installation & Setup

### **1. Clone the Repository**

```bash
git clone [https://github.com/AryaDuhan/mnist_from_scratch.git](https://github.com/AryaDuhan/mnist_from_scratch.git)
cd mnist_from_scratch
```

## Install Dependencies
```bash
pip install numpy
```

## Download MNIST Dataset
- You can download the .idx files from:
- https://www.kaggle.com/datasets/hojjatk/mnist-dataset
- place them inside the mnist_data/ folder:

```bash
mnist_data/
│── train-images.idx3-ubyte
│── train-labels.idx1-ubyte
│── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

## 📊 Example Output
```bash
Starting training on 60000 images for 3 epochs.
Epoch 1/3
Epoch 2/3
Epoch 3/3
Time: 24.84 seconds
Accuracy: 96.31%
```

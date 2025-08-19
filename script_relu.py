import numpy as np
import os
import time
import pickle

# activation functions
def ReLU(x):
    return np.maximum(0, x)

def ReLU_derivative(x):
    return x > 0

def softmax(x):

    # subtract max for numerical stability
    exps = np.exp(x - np.max(x, axis=0))
    return exps / np.sum(exps, axis=0)

# load data
def load_idx_data(images_path, labels_path):
    with open(labels_path, 'rb') as lbpath:
        _, n = np.frombuffer(lbpath.read(8), dtype='>I')
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        _, num, rows, cols = np.frombuffer(imgpath.read(16), dtype='>I')
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(len(labels), 784)
    
    return images, labels

class NeuralNetwork:
    def __init__(self, input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden1_nodes = hidden1_nodes
        self.hidden2_nodes = hidden2_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        # initialize weight and biase
        self.weights_ih1 = np.random.randn(hidden1_nodes, input_nodes) * np.sqrt(2. / input_nodes)
        self.bias_h1 = np.zeros((hidden1_nodes, 1))
        self.weights_h1h2 = np.random.randn(hidden2_nodes, hidden1_nodes) * np.sqrt(2. / hidden1_nodes)
        self.bias_h2 = np.zeros((hidden2_nodes, 1))
        self.weights_h2o = np.random.randn(output_nodes, hidden2_nodes) * np.sqrt(2. / hidden2_nodes)
        self.bias_o = np.zeros((output_nodes, 1))

    def feedforward(self, inputs):
        h1_sum = np.dot(self.weights_ih1, inputs) + self.bias_h1
        h1_output = ReLU(h1_sum)
        h2_sum = np.dot(self.weights_h1h2, h1_output) + self.bias_h2
        h2_output = ReLU(h2_sum)
        final_sum = np.dot(self.weights_h2o, h2_output) + self.bias_o
        final_output = softmax(final_sum)
        return h1_sum, h1_output, h2_sum, h2_output, final_output

    def train(self, inputs, targets):
        h1_sum, h1_output, h2_sum, h2_output, final_outputs = self.feedforward(inputs)

        # backprop
        output_errors = final_outputs - targets
        
        # gradients for output layer
        d_weights_h2o = np.dot(output_errors, h2_output.T)
        d_bias_o = output_errors

        # hidden layer 2 error
        h2_errors = np.dot(self.weights_h2o.T, output_errors) * ReLU_derivative(h2_sum)

        # gradients for hidden layer 2
        d_weights_h1h2 = np.dot(h2_errors, h1_output.T)
        d_bias_h2 = h2_errors

        # hidden layer 1 error
        h1_errors = np.dot(self.weights_h1h2.T, h2_errors) * ReLU_derivative(h1_sum)

        # gradients for hidden layer 1
        d_weights_ih1 = np.dot(h1_errors, inputs.T)
        d_bias_h1 = h1_errors
        
        # update weight and bias
        self.weights_ih1 -= self.learning_rate * d_weights_ih1
        self.bias_h1 -= self.learning_rate * d_bias_h1
        self.weights_h1h2 -= self.learning_rate * d_weights_h1h2
        self.bias_h2 -= self.learning_rate * d_bias_h2
        self.weights_h2o -= self.learning_rate * d_weights_h2o
        self.bias_o -= self.learning_rate * d_bias_o

    def predict(self, inputs):
        *_, final_output = self.feedforward(inputs)
        return np.argmax(final_output)

if __name__ == '__main__':
    path = "mnist_data"
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    X_train, Y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, Y_test = load_idx_data(test_images_path, test_labels_path)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]

    input_nodes = 784
    hidden1_nodes = 128
    hidden2_nodes = 64
    output_nodes = 10
    learning_rate = 0.01
    epochs = 10

    nn = NeuralNetwork(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, learning_rate)

    print(f"\n Starting training on {X_train.shape[0]} images for {epochs} epochs.")
    start_time = time.time()

    for epoch in range(epochs):
        print(f"\n Epoch {epoch+1}/{epochs}")
        for i in range(X_train.shape[0]):
            inputs = X_train[i].reshape(input_nodes, 1)
            targets = np.zeros((output_nodes, 1))
            targets[Y_train[i]] = 1
            nn.train(inputs, targets)

            if (i + 1) % 10000 == 0:
                print(f"Processed {i+1}/{X_train.shape[0]} images")

    end_time = time.time()
    print(f"\n Training Complete! Took {end_time - start_time:.2f} seconds.")

    print("\nTesting...")
    correct_predictions = 0
    for i in range(len(Y_test)):
        inputs = X_test[i].reshape(input_nodes, 1)
        label = Y_test[i]
        prediction = nn.predict(inputs)
        if prediction == label:
            correct_predictions += 1

    accuracy = (correct_predictions / len(Y_test)) * 100
    print(f"\n Test accuracy: {accuracy:.2f}%")

    print("\nSaving the model...")
    with open('trained_model_relu.pkl', 'wb') as f:
        pickle.dump(nn, f)
    print("Model saved successfully.")
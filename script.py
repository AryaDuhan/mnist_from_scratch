import numpy as np
import os
import time


def sigmoid(x):
    x=np.clip(x, -500,500)
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

def load_idx_data(images_path, labels_path):
    # open file in read binary(rb) mode
    with open(labels_path, 'rb') as lbpath:

        #reads the 8-byte header of the label file
        #frombuffer takes raw byte data and interprets it as numbers
        #dtype=>I tells numpy to interpret the bytes as 4-bit unsigned integer
        # _ is used for varibale we dont care about, n holds the number of labels
        _, n =np.frombuffer(lbpath.read(8), dtype='>I')

        #reads all remaining bytes in the file and interprets each byte as 8-bit unsigned integer (0 to 255)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)
    
    with open(images_path, 'rb') as imgpath:
        #same logic but reads the 16 bytes header of the file
        _, num, rows,cols=np.frombuffer(imgpath.read(16), dtype='>I')

        #reshape data in useful format of a 2d array
        #each row corresponds to one image with 784 pixels
        images=np.frombuffer(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    
    return images,labels


class NeuralNetwork:

    #creates a new instance of NeuralNetwork
    def __init__(self,input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        #xavier initialization
        self.weights_ih = np.random.randn(self.hidden_nodes, self.input_nodes) * np.sqrt(2. / (self.input_nodes + self.hidden_nodes))
        self.weights_ho = np.random.randn(self.output_nodes, self.hidden_nodes) * np.sqrt(2. / (self.hidden_nodes + self.output_nodes))

        self.bias_h = np.zeros((self.hidden_nodes, 1))
        self.bias_o = np.zeros((output_nodes, 1))



    def feedforward(self,inputs):
        #zh= wih . x + bih
        hidden_sum = np.dot(self.weights_ih, inputs) + self.bias_h

        #Ah=sigma(zh)
        hidden_output = sigmoid(hidden_sum)
        
        #zo= who . Ah + bho
        final_sum=np.dot(self.weights_ho, hidden_output) + self.bias_o

        #Ao=sigma(zo)
        final_output = sigmoid(final_sum)

        return hidden_output, final_output
    


    def train(self, inputs, targets):
        hidden_outputs,final_outputs = self.feedforward(inputs)

        # errorO= y - Ao
        output_errors=targets-final_outputs
        # sigma'(zo) * y-Ao * alpha
        output_gradient=sigmoid_derivative(final_outputs)*output_errors*self.learning_rate

        # sigma'(zo) * y-Ao * alpha * Ah.T
        delta_weights_ho=np.dot(output_gradient, hidden_outputs.T)
        # who=who + alpha(del sigma/ del who)
        # where (del sigma/ del who) = sigma'(zo) * y-Ao * Ah.T
        self.weights_ho += delta_weights_ho
        # bho= bho + sigma'(zo) * y-Ao * alpha
        self.bias_o+=output_gradient

        # errorH= Who.T . y - Ao 
        hidden_errors=np.dot(self.weights_ho.T, output_errors)
        # sigma'(zh) * Who.T . y - Ao * alpha
        hidden_gradient=sigmoid_derivative(hidden_outputs)*hidden_errors*self.learning_rate

        # (sigma'(zh) * Who.T . y - Ao * alpha). x.T
        delta_weights_ih=np.dot(hidden_gradient, inputs.T)
        # wih=wih + alpha(del sigma/ del wih)
        self.weights_ih+=delta_weights_ih
        # bih = bih + sigma'(zh) * y-Ao * alpha
        self.bias_h+=hidden_gradient


    def predict(self,inputs):
        # feedforward returns Ah and Ao, we discard the Ah cause we don't need them
        _,final_output = self.feedforward(inputs)
        # Ao is a vector of 10 numbers, gives the best possiblity and match
        return np.argmax(final_output)
    


if __name__ == '__main__':
    path = "mnist_data"

    #load the training data
    train_images_path = os.path.join(path, 'train-images.idx3-ubyte')
    train_labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
    
    #load the test data
    test_images_path = os.path.join(path, 't10k-images.idx3-ubyte')
    test_labels_path = os.path.join(path, 't10k-labels.idx1-ubyte')

    #split the train and test data
    X_train, Y_train = load_idx_data(train_images_path, train_labels_path)
    X_test, Y_test = load_idx_data(test_images_path, test_labels_path)

    #normalization from (0,255) to (0,1.0)
    X_train = X_train /255.0
    X_test = X_test/255.0

    #shuffling
    permutation = np.random.permutation(X_train.shape[0])
    X_train = X_train[permutation]
    Y_train = Y_train[permutation]

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.1
    epochs = 3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)



print(f"\n Starting training on {X_train.shape[0]} images for {epochs} epochs.")
start_time = time.time()

for epoch in range(epochs):
    print(f"\n Epoch {epoch+1}/{epochs}")

    # loop of every image in array
    for i in range (X_train.shape[0]):

        #convert the image into a matrix (784 , 1)
        inputs = X_train[i].reshape(input_nodes,1)

        #creates vector of 10 zeros and then place 1 at the index of the correct digit
        targets = np.zeros((output_nodes,1))
        targets[Y_train[i]]=1

        nn.train(inputs, targets)

        # print after every 10k images
        if(i+1)%10000 ==0:
            print(f"Processed {i+1}/{X_train.shape[0]} images")


end_time = time.time()
print(f"\n Training Complete! Took {end_time - start_time:.2f} seconds.")

print("Testing")
correct_predictions = 0

for i in range(len(Y_test)):
    inputs = X_test[i].reshape(input_nodes,1)
    label = Y_test[i]

    prediction = nn.predict(inputs)

    if prediction == label:
        correct_predictions +=1 

accuracy = (correct_predictions/ len(Y_test)) *100
print(f"\n Test accuracy: {accuracy:.2f}%")
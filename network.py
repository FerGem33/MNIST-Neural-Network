from time import time
import numpy as np
from util.functions import funcs


class Layer:
    def __init__(self, n_in: int, n_out: int, activation):
        """
        Single layer of a neural network.
        :param n_in: number of inputs (neurons in previous layer)
        :param n_out: number of outputs (neurons in layer)
        :param activation: activation function for the layer
        """
        self.n = n_out  # neurons in layer
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)  # weights
        self.b = np.zeros((1, n_out))  # biases
        self.activation = activation

        self.a = None  # activations
        self.z = None  # weighted sum

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass of the layer.
        :param x: activation of the previous layer
        :return: activation of the layer
        """
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a


class NeuralNetwork:
    def __init__(self):
        """
        Neural network for classification.
        """
        self.l = []
        self.L = 0

    def save_network(self, filename):
        """
        Saves the weights and biases of the model.
        :param filename: name of the file to save to.
        """
        weights = []
        biases = []

        for layer in self.l:
            weights.append(layer.W)
            biases.append(layer.b)
        params = {}

        for i, (w, b) in enumerate(zip(weights, biases)):
            params[f'W{i}'] = w
            params[f'b{i}'] = b

        np.savez(f'models/{filename}.npz', **params)

    def load_network(self, filename):
        """
        Loads the weights and biases of an existing model.
        :param filename: name (without extension) of the file to load.
        """
        self.__init__()
        params = np.load(f'models/{filename}.npz')
        L = len(params) // 2

        for i in range(L):
            W = params[f'W{i}']
            b = params[f'b{i}']
            self.add_layer(W.shape[0], W.shape[1], "relu")
            self.l[i].W = W
            self.l[i].b = b
        self.l[-1].activation = funcs["softmax"]

    def add_layer(self, n_in: int, n_out: int, activation: str):
        """
        Adds a new layer at the end of the network.
        :param n_in: number of inputs
        :param n_out: number of outputs (neurons in layer)
        :param activation: activation function for the layer
        """
        self.l.append(Layer(n_in, n_out, funcs[activation]))
        self.L += 1

    def forward(self, x: np.ndarray):
        """
        Performs the forward propagation on the network.
        :param x: input data for the first layer
        """
        a = x
        for layer in range(self.L):
            a = self.l[layer].forward(a)

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float, batch_size: int):
        """
        Performs the backpropagation of the network using gradient descent algorithm.
        :param x: input data for the first layer
        :param y: expected output, must be one-hot encoded
        :param lr: learning rate
        :param batch_size: size of the batches used during training
        """
        # Initialize accumulators for weights and bias gradients
        dW = [np.zeros_like(layer.W) for layer in self.l]
        db = [np.zeros_like(layer.b) for layer in self.l]

        # Initialize the gradient of last layer
        delta = self.l[-1].a - y    # dL/dz for the output layer

        # Loop through layers in reverse order (starting from the last layer)
        for i in reversed(range(1, self.L)):
            # Add the gradients of weights and biases
            dW[i] += np.dot(self.l[i-1].a.T, delta)
            db[i] += np.sum(delta, axis=0, keepdims=True)

            # Calculate the gradient for the next layer
            delta = np.dot(delta, self.l[i].W.T) * (self.l[i - 1].z > 0)   # dL/dz for the next layer

        # Add the gradients of weights and bias of the first layer
        dW[0] += np.dot(x.T, delta)
        db[0] += np.sum(delta, axis=0, keepdims=True)

        for i in range(self.L):
            self.l[i].W -= lr * (dW[i] / batch_size)
            self.l[i].b -= lr * (db[i] / batch_size)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float, batch_size: int = 20):
        """
        Handles the whole cycle of propagation and backpropagation of the network.
        :param X: input data for the first layer
        :param Y: expected output, must be one-hot encoded
        :param epochs: number of epochs
        :param lr: learning rate
        :param batch_size: size of the batches used during training
        """
        t = X.shape[0]
        batches = t // batch_size
        for epoch in range(epochs):
            # Shuffle dataset
            shuffle = np.random.permutation(t)
            x = X[shuffle]
            y = Y[shuffle]

            epoch_loss = 0
            t1 = time()

            for i in range(batches):
                x_i = x[i * batch_size: (i + 1) * batch_size]
                y_i = y[i * batch_size: (i + 1) * batch_size]
                self.forward(x_i)
                self.backward(x_i, y_i, lr, batch_size)
                loss = funcs["cross_entropy"](self.l[-1].a, y_i)
                epoch_loss += loss
            epoch_loss /= t

            print(f' epoch: {epoch}, average loss: {epoch_loss:.6f}, time: {time() - t1:.3f}s')

    def predict(self, X: np.ndarray, Y: np.ndarray, batch_size: int = 20):
        """
        Performs the forward propagation without training the network.
        :param X: input data for the first layer
        :param Y: expected output, must be one-hot encoded
        :param batch_size: size of the batches used during training
        """
        t = X.shape[0]
        batches = t // batch_size

        loss = 0
        predicted = 0
        t1 = time()

        for i in range(batches):
            x_i = X[i * batch_size: (i + 1) * batch_size]
            y_i = Y[i * batch_size: (i + 1) * batch_size]
            self.forward(x_i)
            loss += funcs["cross_entropy"](self.l[-1].a, y_i) * batch_size
            predicted += np.sum(np.argmax(self.l[-1].a, axis=1) == np.argmax(y_i, axis=1))
        loss /= t
        accuracy = predicted / t

        print(f'average loss: {loss:.6f}, predicted: {predicted}, Accuracy: {accuracy:.5f}, time: {time() - t1:.3f}s')

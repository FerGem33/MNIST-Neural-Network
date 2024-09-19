from time import time
import numpy as np

epsilon = 1e-8  # used to guarantee numerical stability

# activation functions
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=1, keepdims=True)
    z = np.exp(z - z_max)
    return z / (z.sum() + epsilon)


# loss function
def cross_entropy(a: np.ndarray, y: np.ndarray):
    a = np.clip(a, epsilon, 1 - epsilon)
    i = np.argmax(y[0])
    return - np.log(a[0][i])


class Layer:
    def __init__(self, n_in: int, n_out: int, activation):
        self.n = n_out  # neurons in layer
        self.W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)  # weights
        self.b = np.zeros((1, n_out))  # biases
        self.activation = activation

        self.a = np.empty((1, n_out))  # activations
        self.z = np.empty((1, n_out))  # weighted sum

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.z = np.dot(x, self.W) + self.b
        self.a = self.activation(self.z)
        return self.a


class NeuralNetwork:
    def __init__(self):
        self.l = []
        self.L = 0

    def save_network(self, filename):
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
        self.__init__()
        params = np.load(f'models/{filename}.npz')
        L = len(params) // 2

        for i in range(L):
            W = params[f'W{i}']
            b = params[f'b{i}']
            self.add_layer(W.shape[0], W.shape[1], relu)
            self.l[i].W = W
            self.l[i].b = b
        self.l[-1].activation = softmax

    def add_layer(self, n_in: int, n_out: int, activation):
        self.l.append(Layer(n_in, n_out, activation))
        self.L += 1

    def forward(self, x: np.ndarray):
        a = x
        for layer in range(self.L):
            a = self.l[layer].forward(a)

    def backward(self, x: np.ndarray, y: np.ndarray, lr: float):
        # Initialize the gradient for the last layer (output layer)
        delta = self.l[-1].a - y  # dL/dz for the output layer

        # Loop through layers in reverse order (starting from the last layer)
        for i in reversed(range(1, self.L)):
            # Gradient of weights and biases
            self.l[i].W -= lr * np.dot(self.l[i - 1].a.T, delta)
            self.l[i].b -= lr * np.sum(delta, axis=1, keepdims=True)

            # Calculate the gradient for the next layer
            delta = np.dot(delta, self.l[i].W.T) * (self.l[i - 1].z > 0)  # dL/dz for the next layer

        # Update the weights and biases for the first layer
        self.l[0].W -= lr * np.dot(x.T, delta)
        self.l[0].b -= lr * np.sum(delta, axis=1, keepdims=True)

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float = 0.0001):
        t = len(X)
        for epoch in range(epochs):
            # Shuffle dataset
            shuffle = np.random.permutation(t)
            x = X[shuffle]
            y = Y[shuffle]

            epoch_loss = 0
            t1 = time()

            for i in range(t):
                x_i = np.array([x[i]])
                y_i = np.array([y[i]])
                self.forward(x_i)
                self.backward(x_i, y_i, lr)
                loss = cross_entropy(self.l[-1].a, y_i)
                epoch_loss += loss

            print(f' epoch: {epoch}, Average loss: {epoch_loss / t}, Time: {time() - t1}s')

    def predict(self, X: np.ndarray, Y: np.ndarray):
        t = len(X)
        loss = 0
        predicted = 0
        t1 = time()

        for i in range(t):
            x_i = np.array([X[i]])
            y_i = np.array([Y[i]])
            self.forward(x_i)
            loss += cross_entropy(self.l[-1].a, y_i)
            if np.argmax(self.l[-1].a) == np.argmax(y_i):
                predicted += 1

        print(f'Average loss: {loss / t}, Predicted: {predicted}, Accuracy: {predicted / t}, Time: {time() - t1}s')

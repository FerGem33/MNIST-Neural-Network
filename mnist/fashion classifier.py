import numpy as np
import pandas as pd
import ctypes
from network import NeuralNetwork, relu, softmax

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# load datasets
train = pd.read_csv('data/fashion/fashion-mnist_train.csv')
test = pd.read_csv('data/fashion/fashion-mnist_test.csv')


temp = train.iloc[:, 0].values.astype(int)
train_labels = np.zeros(shape=(60000, 10))
for i in range(60000):
    train_labels[i][temp[i]] = 1
train_data = train.iloc[:, 1:].values / 255

temp = test.iloc[:, 0].values.astype(int)
test_labels = np.zeros(shape=(60000, 10))
for i in range(10000):
    test_labels[i][temp[i]] = 1
test_data = test.iloc[:, 1:].values / 255


# Prevent system sleep
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

# create network
nn = NeuralNetwork()


"""
nn.add_layer(784, 128, relu)
nn.add_layer(128, 10, softmax)
"""

nn.load_network('nn2')

# train network
nn.train(train_data, train_labels, 15, 0.0001)

# validation test
permutation = np.random.permutation(len(test_data))
nn.predict(test_data[permutation], test_labels[permutation])

# save weights and biases
# nn.save_network('nn2')

# Reset to allow sleep
ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

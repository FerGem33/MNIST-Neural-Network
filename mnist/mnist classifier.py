import numpy as np
import pandas as pd

from network import NeuralNetwork, relu, softmax

# import mnist dataset
data = np.array(pd.read_csv('data/train.csv')) / 255
data_labels = np.array(pd.read_csv('data/train_labels.csv'))
test = np.array(pd.read_csv('data/test.csv')) / 255
test_labels = np.array(pd.read_csv('data/test_labels.csv'))

# create network
nn = NeuralNetwork()
nn.add_layer(784, 128, relu)
nn.add_layer(128, 10, softmax)

# train network
nn.train(data, data_labels, epochs=20)

# validation test
permutation = np.random.permutation(len(test))
nn.predict(test[permutation], test_labels[permutation])

# save weights and biases
nn.save_network('nn1')

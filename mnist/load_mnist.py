import numpy as np
from scipy.io import loadmat
from pandas import DataFrame

""" 
Note:
The file 'data/mnist-original.mat' is not included because of size constraints, but
it can be found on Kaggle: https://www.kaggle.com/datasets/hojjatk/mnist-dataset 
"""

mnist = loadmat('data/mnist-original.mat')

data = mnist['data'].T
label = mnist['label']

label = label.astype(int)
labels = np.zeros(shape=(70000, 10))
for i in range(0, 70000):
    labels[i][label[0][i]] = 1

train = DataFrame(data[:60000])
train.to_csv('data/train.csv', index=False)
test = DataFrame(data[60000:])
test.to_csv('data/test.csv', index=False)

train_labels = DataFrame(labels[:60000])
train_labels.to_csv('data/train_labels.csv', index=False)
test_labels = DataFrame(labels[60000:])
test_labels.to_csv('data/test_labels.csv', index=False)

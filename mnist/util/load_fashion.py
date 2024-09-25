import numpy as np
from pandas import DataFrame, read_csv

""" 
Note:
The files 'fashion-mnist_train.csv' and 'fashion-mnist_test.csv'
can be found on Kaggle: 'https://www.kaggle.com/datasets/zalando-research/fashionmnist'
"""


def prepare_dataset():
    train = read_csv('../data/fashion/fashion-mnist_train.csv')
    test = read_csv('../data/fashion/fashion-mnist_test.csv')

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

    DataFrame(train_data).to_csv('../data/fashion/train.csv', index=False)
    DataFrame(train_labels).to_csv('../data/fashion/train_labels.csv', index=False)
    DataFrame(test_data).to_csv('../data/fashion/test.csv', index=False)
    DataFrame(test_labels).to_csv('../data/fashion/test_labels.csv', index=False)


def load_train():
    data = np.array(read_csv('data/fashion/train.csv'))
    data_labels = np.array(read_csv('data/fashion/train_labels.csv'))
    return data, data_labels


def load_test():
    test = np.array(read_csv('data/fashion/test.csv'))
    test_labels = np.array(read_csv('data/fashion/test_labels.csv'))
    return test, test_labels


if __name__ == "__main__":
    prepare_dataset()

import numpy as np
from pandas import read_csv
from mnist.network import NeuralNetwork
from mnist.util.functions import funcs
import ctypes


class NetworkRunner:
    def __init__(self, name: str):
        self.name = name
        self.nn = NeuralNetwork()

    def load_train(self):
        data = np.array(read_csv(f'data/{self.name}/train.csv'))
        data_labels = np.array(read_csv(f'data/{self.name}/train_labels.csv'))
        return data, data_labels

    def load_test(self):
        test = np.array(read_csv(f'data/{self.name}/test.csv'))
        test_labels = np.array(read_csv(f'data/{self.name}/test_labels.csv'))
        return test, test_labels
    
    def create_network(self):
        self.nn.l = []

        print("Available functions: ")
        for func in funcs:
            print(f'[{func}] ')
        print()

        i = 1
        n = [int(input("Input size of layer 1?: "))]

        while True:
            n.append(int(input(f"Neurons of layer {i}?: ")))
            f = input("Activation function? ([softmax] to finish): ")

            self.nn.add_layer(n[i-1], n[i], f)
            i += 1

            if f == "softmax":
                break
        print()

    def run(self):
        # Prevent system sleep
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

        data, data_labels = self.load_train()
        test, test_labels = self.load_test()
        filename = ''
    
        create_new = True if input("Create new network? (y/n): ") == "y" else False
        if create_new:
            self.create_network()
        else:
            filename = input(f'Name of file to load from? (defaults to \"{self.name}\"): ')
            if filename == '':
                filename = self.name
            self.nn.load_network(filename)
            print(f"Loaded network from \"models/{filename}.npz\"")
            print()

        train = True if input("Train? (y/n): ") == "y" else False
        epochs = 0
        if train:
            epochs = int(input("Number of epochs?: "))
            print()

        save = True if input("Save network? (y/n): ") == "y" else False
        if save:
            temp = input(f'Name of file to save to? (defaults to \"{filename}\"): ')
            if temp != '':
                filename = temp
        print()

        if train:
            print("Training")
            self.nn.train(data, data_labels, epochs, 0.001)
            print()

        # validation test
        print("Validation test")
        self.nn.predict(test, test_labels)
        print()

        if save:
            self.nn.save_network(filename)
            print(f"Saved network to \"models/{filename}.npz\"")
            print()

        # Reset to allow sleep
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

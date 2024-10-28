# MNIST Neural Network
An artificial neural network for image classification implemented from scratch, using no machine learning frameworks but basic Python libraries such as NumPy, SciPy and Pandas. 

The network uses ReLU as activation for input and hidden layers and Softmax for the output layer,  the training is performed with the batch gradient descent algorithm. This repository comes with pre-trained models, but different networks can be created varying the number of layers and neurons per layer.

The **mnist** dataset contains  images of handwritten digits and the **fashion-mnist** contains images of clothes and accessories. Both datasets have 70k gray-scale images of size 28x28 px, where 60k images are intended for training and 10k for testing.

## Models
The following pre-trained models are available:

### mnist

| Model  | Description                                 | Accuracy | Epochs |
|--------|---------------------------------------------|----------|--------|
| mnist  | relu(128), relu(64), softmax(10)            | 96.26%   | 100    |
| mnist1 | relu(128), relu(64), softmax(10)            | 96.11%   | 50     |
| mnist2 | relu(392), relu(196), relu(49), softmax(10) | 96.08%   | 50     |

### fashion

| Model    | Description                                 | Accuracy | Epochs |
|----------|---------------------------------------------|----------|--------|
| fashion  | relu(128), relu(64), softmax(10)            | 88.51%   | 100    |
| fashion1 | relu(128), relu(64), softmax(10)            | 87.04%   | 50     |
| fashion2 | relu(392), relu(196), relu(49), softmax(10) | 88.52%   | 50     |


The description indicates the activation function and the number of neurons of each layer. The input size of all models is 784.

## Setup
To run the models you need to have python installed on your machine and complete the following steps:

First install the needed dependencies:
````
pip install -r requirements.txt
````

You also need to download the data and preprocess it, both datasets can be found on Kaggle:
- [mnist](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [fashion mnist](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

Download the datasets in zip format and extract them into the data folder, make sure you have these files directly placed in the data folder:
- `mnist-original.mat`
- `fashion-mnist_train.csv`
- `fashion-mnist_test.csv`

After that, run the following command on the terminal:
```
python src/util/setup.py
```
Now you are ready to use the model.

## Usage

Execute the files `mnist.py` or `fashion.py` depending on which dataset you wish to work with:
````
python mnist.py
````

or 

````
python fashion.py
````

import numpy as np

epsilon = 1e-8  # used to guarantee numerical stability


# activation functions
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    z_max = np.max(z, axis=1, keepdims=True)
    z = np.exp(z - z_max)
    return z / (np.sum(z, axis=1, keepdims=True) + epsilon)


# loss function
def cross_entropy(a: np.ndarray, y: np.ndarray):
    a = np.clip(a, epsilon, 1 - epsilon)
    return -np.sum(y * np.log(a), axis=1).mean()


funcs = {
    "relu": relu,
    "softmax": softmax,
    "cross_entropy": cross_entropy,
}

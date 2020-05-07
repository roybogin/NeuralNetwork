from network_model import NetworkModel
from typing import List
from activation import Activation, Sigmoid, Tanh
from losses import Loss, MSE
import numpy as np


class NeuralNetwork(NetworkModel):
    def __init__(self, layers_num: List[int], activations: List[Activation], loss_func: Loss, lr: float, min_init_weight=-0.5, max_init_weight=0.5):
        self.layer_weights = []  # weight arrays for each layer: weights[n] is weights between n and n+1 layer
        self.biases = []  # bias arrays for each layer: biases[n] is biases for n+1 layer
        self.activations = activations
        self.loss_function = loss_func
        self.lr = lr
        self.layers_numbers = layers_num
        for i in range(len(layers_num) - 1):
            self.layer_weights.append((max_init_weight - min_init_weight) * np.random.rand(layers_num[i], layers_num[i + 1]) + min_init_weight)
            self.biases.append((max_init_weight - min_init_weight) * np.random.rand(layers_num[i+1]) + min_init_weight)

    def predict(self, data: np.ndarray):
        for i in range(len(self.layer_weights)):
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        return data

    def train_vanish_grad(self, my_input: np.ndarray, labels: np.ndarray):
        data = my_input
        layer_output = []
        for i in range(len(self.layer_weights)):
            layer_output.append(data)
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        # data is output now

        error_deriv = self.loss_function.derivative(data, labels)
        deltas = [self.activations[-1].derivative(data) * error_deriv]
        for i in range(len(self.layer_weights) - 1, 0, -1):
            deltas.append(self.activations[i].derivative(layer_output[i]) * np.dot(deltas[-1], self.layer_weights[i].T))

        deltas.reverse()
        for i in range(len(deltas)):
            self.layer_weights[i] = self.layer_weights[i] - self.lr * np.sign(np.dot(layer_output[i].T, deltas[i]))
            self.biases[i] = self.biases[i] - self.lr * np.sign(np.sum(deltas[i], axis=0))

    def train(self, my_input: np.ndarray, labels: np.ndarray):
        data = my_input
        layer_outputs = []
        for i in range(len(self.layer_weights)):
            layer_outputs.append(data)
            data = self.activations[i].calculate(np.dot(data, self.layer_weights[i]) + self.biases[i])

        error_deriv = self.loss_function.derivative(data, labels)
        deltas = [self.activations[-1].derivative(data) * error_deriv]
        for i in range(len(self.layer_weights) - 1, 0, -1):
            deltas.append(self.activations[i].derivative(layer_outputs[i]) * np.dot(deltas[-1], self.layer_weights[i].T))

        deltas.reverse()
        for i in range(len(deltas)):
            self.layer_weights[i] = self.layer_weights[i] - self.lr * np.dot(layer_outputs[i].T, deltas[i])
            self.biases[i] = self.biases[i] - self.lr * np.sum(deltas[i], axis=0)

    def save_weights(self, path: str, weights_file: str, bias_file: str):
        np.save(path + "/" + weights_file, self.layer_weights, allow_pickle=True)
        np.save(path + "/" + bias_file, self.biases, allow_pickle=True)

    def load_weights(self, path: str, weights_file: str, bias_file: str):
        self.layer_weights = np.load(path + "/" + weights_file, allow_pickle=True)
        self.biases = np.load(path + "/" + bias_file, allow_pickle=True)


def main():
    nn = NeuralNetwork([2, 4, 1], [Tanh, Tanh], MSE, 1)
    data = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    labels = np.array([[-1, 1, 1, -1]]).T
    print(nn.predict(data))
    for i in range(10000):
        nn.train(data, labels)
    print(nn.predict(data))


if __name__ == "__main__":
    main()


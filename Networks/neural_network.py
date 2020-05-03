from typing import List
from activation import Activation, Sigmoid
from losses import SSE
import numpy as np


class NeuralNetwork:
    def __init__(self, layers_num: List[int], activations: List[Activation], lr=0.01):
        assert len(activations) == len(layers_num) - 1
        self.layer_weights = []  # weight arrays for each layer: weights[n] is weights between n and n+1 layer
        self.biases = []  # bias arrays for each layer: biases[n] is biases for n+1 layer
        self.activations = activations
        self.lr = lr
        self.layers_numbers = layers_num
        for i in range(len(layers_num) - 1):
            self.layer_weights.append(np.random.rand(layers_num[i], layers_num[i + 1]))
            self.biases.append(np.random.rand(layers_num[i+1]))

    def predict(self, data: np.ndarray):
        for i in range(len(self.layer_weights)):
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        return data

    def train_vanish_grad(self, data: np.ndarray, labels: np.ndarray):
        layer_output = []
        for i in range(len(self.layer_weights)):
            layer_output.append(data)
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        # data is output now

        error_deriv = SSE.derivative(data, labels)
        deltas = [self.activations[-1].derivative(data) * error_deriv]
        for i in range(len(self.layer_weights) - 1, 0, -1):
            deltas.append(self.activations[i].derivative(layer_output[i]) * np.dot(deltas[-1], self.layer_weights[i].T))

        deltas.reverse()
        for i in range(len(deltas)):
            self.layer_weights[i] = self.layer_weights[i] - self.lr * np.sign(np.dot(layer_output[i].T, deltas[i]))
            self.biases[i] = self.biases[i] - self.lr * np.sign(np.sum(deltas[i], axis=0))

    def train(self, data: np.ndarray, labels: np.ndarray):
        layer_output = []
        for i in range(len(self.layer_weights)):
            layer_output.append(data)
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        # data is output now

        error_deriv = SSE.derivative(data, labels)
        deltas = [self.activations[-1].derivative(data) * error_deriv]
        for i in range(len(self.layer_weights) - 1, 0, -1):
            deltas.append(self.activations[i].derivative(layer_output[i]) * np.dot(deltas[-1], self.layer_weights[i].T))

        deltas.reverse()
        for i in range(len(deltas)):
            self.layer_weights[i] = self.layer_weights[i] - self.lr * np.dot(layer_output[i].T, deltas[i])
            self.biases[i] = self.biases[i] - self.lr * np.sum(deltas[i], axis=0)

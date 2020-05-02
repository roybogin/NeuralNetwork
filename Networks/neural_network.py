from typing import List
from activation import Activation, Sigmoid
from losses import MSE
import numpy as np


class NeuralNetwork:
    def __init__(self, layers_num: List[int], activations: List[Activation], lr=0.01):
        assert len(activations) == len(layers_num) - 1
        self.layer_weights = []  # weight arrays for each layer: weights[n] is weights between n and n+1 layer
        self.biases = []  # bias arrays for each layer: biases[n] is biases for n+1 layer
        self.activations = activations
        self.lr = lr
        for i in range(len(layers_num) - 1):
            self.layer_weights.append(np.random.rand(layers_num[i], layers_num[i + 1]))
            self.biases.append(np.random.rand(layers_num[i+1]))

    def predict(self, data: np.ndarray):
        for i in range(len(self.layer_weights)):
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        return data

    def train(self, data: np.ndarray, labels: np.ndarray):
        layer_inputs = []
        for i in range(len(self.layer_weights)):
            layer_inputs.append(data)
            data = data.dot(self.layer_weights[i]) + self.biases[i]
            data = self.activations[i].calculate(data)
        # data is output now

        error_deriv = MSE.derivative(data, labels)
        deltas = [self.activations[-1].derivative(data) * error_deriv]
        for i in range(len(self.layer_weights) - 1, 0, -1):
            deltas.append(self.activations[i].derivative(layer_inputs[i]) * np.dot(deltas[-1], self.layer_weights[i].T))

        deltas.reverse()
        for i in range(len(deltas)):
            self.layer_weights[i] = self.layer_weights[i] + self.lr * np.dot(layer_inputs[i].T, deltas[i])
            self.biases[i] = self.biases[i] + np.sum(deltas[i], axis=0)

def main():
    nn = NeuralNetwork((2, 4, 1), [Sigmoid, Sigmoid], lr=1.0)
    data = np.array([[0, 0],
                     [0, 1],
                     [1, 0]]).astype(float)
    label = np.array([[0, 1, 1]]).T.astype(float)
    print(nn.predict(data))
    for i in range(10000):
        nn.train(data, label)

if __name__ == '__main__':
    main()
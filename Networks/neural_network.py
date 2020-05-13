from network_model import NetworkModel
from typing import List
from activation import Activation, Sigmoid, Tanh, Linear
from losses import Loss, MSE
import numpy as np
from layer import Dense

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
            data = Dense.feed_forward(data, self.layer_weights[i], self.biases[i], self.activations[i])
        return data

    def train_vanish_grad(self, my_input: np.ndarray, labels: np.ndarray):
        data = my_input
        layer_outputs = []
        for i in range(len(self.layer_weights)):
            layer_outputs.append(data)
            data = Dense.feed_forward(data, self.layer_weights[i], self.biases[i], self.activations[i])

        for i in range(len(self.layer_weights), 0, -1):
            if i == len(self.layer_weights):
                delta = Dense.back_propagate_last_layer(data, labels, self.layer_weights[i - 1], self.biases[i - 1],
                                                        layer_outputs[i - 1], self.loss_function, self.activations[-1],
                                                        self.lr, vanish_grad=True)
            else:
                delta = Dense.back_propagate(delta, layer_outputs[i], self.layer_weights[i], self.layer_weights[i - 1],
                                             self.biases[i - 1], layer_outputs[i - 1], self.activations[i - 1], self.lr, vanish_grad=True)

    def train(self, my_input: np.ndarray, labels: np.ndarray):
        data = my_input
        layer_outputs = []
        for i in range(len(self.layer_weights)):
            layer_outputs.append(data)
            data = Dense.feed_forward(data, self.layer_weights[i], self.biases[i], self.activations[i])

        for i in range(len(self.layer_weights), 0, -1):
            if i == len(self.layer_weights):
                delta = Dense.back_propagate_last_layer(data, labels, self.layer_weights[i-1], self.biases[i-1], layer_outputs[i-1], self.loss_function, self.activations[-1], self.lr)
            else:
                delta = Dense.back_propagate(delta, layer_outputs[i], self.layer_weights[i], self.layer_weights[i-1], self.biases[i-1], layer_outputs[i-1], self.activations[i-1], self.lr)

    def save_weights(self, path: str, weights_file: str, bias_file: str):
        np.save(path + "/" + weights_file, self.layer_weights, allow_pickle=True)
        np.save(path + "/" + bias_file, self.biases, allow_pickle=True)

    def load_weights(self, path: str, weights_file: str, bias_file: str):
        self.layer_weights = np.load(path + "/" + weights_file, allow_pickle=True)
        self.biases = np.load(path + "/" + bias_file, allow_pickle=True)


def main():
    nn = NeuralNetwork([2, 4, 1], [Tanh, Tanh], MSE, 0.1)
    data = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    labels = np.array([[-1, 1, 1, -1]]).T
    print(nn.predict(data))
    for i in range(20000):
        nn.train(data, labels)
    print(nn.predict(data))


if __name__ == "__main__":
    main()

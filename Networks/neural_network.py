from network_model import NetworkModel
from typing import List
from activation import Activation, Sigmoid, Tanh, Linear
from losses import Loss, MSE
import numpy as np
from layer import Dense


class NeuralNetwork(NetworkModel):
    def __init__(self, layers_num: List[int], activations: List[Activation], loss_func: Loss, lr: float, min_init_weight=-0.5, max_init_weight=0.5):
        self.layers = []
        for i in range(len(layers_num) - 1):
            self.layers.append(Dense(layers_num[i], layers_num[i+1], activations[i]))
        self.loss_function = loss_func
        self.lr = lr
        self.layers_numbers = layers_num

    def predict(self, data: np.ndarray):
        for layer in self.layers:
            data = layer.feed_forward(data)
        return data

    def train(self, my_input: np.ndarray, labels: np.ndarray, vanish_grad=False):
        data = my_input
        layer_outputs = []
        for layer in self.layers:
            layer_outputs.append(data)
            data = layer.feed_forward(data)

        for i in range(len(self.layers), 0, -1):
            if i == len(self.layers):
                delta = self.layers[i-1].back_propagate_last_layer(data, labels, layer_outputs[i-1], self.loss_function, self.lr, vanish_grad)
            else:
                delta = self.layers[i-1].back_propagate(delta, layer_outputs[i], self.layers[i].get_weights(), layer_outputs[i-1], self.lr, vanish_grad)

    def save_weights(self, path: str, weights_file: str, bias_file: str):
        all_weights = []
        all_biases = []
        for layer in self.layers:
            all_weights.append(layer.get_weights())
            all_biases.append(layer.get_biases())
        np.save(path + "/" + weights_file, all_weights, allow_pickle=True)
        np.save(path + "/" + bias_file, all_biases, allow_pickle=True)

    def load_weights(self, path: str, weights_file: str, bias_file: str):
        all_weights = np.load(path + "/" + weights_file, allow_pickle=True)
        all_biases = np.load(path + "/" + bias_file, allow_pickle=True)
        for layer in range(len(self.layers)):
            self.layers[layer].set_weights(all_weights[layer])
            self.layers[layer].set_biases(all_biases[layer])


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

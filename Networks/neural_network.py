from network_model import NetworkModel
from typing import List
from activation import Activation
from losses import Loss
import numpy as np
from layer import Dense


class NeuralNetwork(NetworkModel):
    def __init__(self, layers_num: List[int], activations: List[Activation], loss_func: Loss, lr: float, min_init_weight=-0.5, max_init_weight=0.5):
        self.layers = []
        for i in range(len(layers_num) - 1):
            self.layers.append(Dense(layers_num[i], layers_num[i+1], activations[i], min_init_weight=min_init_weight, max_init_weight=max_init_weight))
        self.loss_function = loss_func
        self.lr = lr
        self.layers_numbers = layers_num

    def predict(self, data: np.ndarray):
        for layer in self.layers:
            data = layer.feed_forward(data)
        return data

    def train(self, net_input: np.ndarray, labels: np.ndarray):
        data = net_input
        layer_outputs = []
        for layer in self.layers:
            # feed forward for layers
            layer_outputs.append(data)
            data = layer.feed_forward(data)

        for i in range(len(self.layers), 0, -1):
            # back propagate from last layer to first
            if i == len(self.layers):
                delta = self.layers[i-1].back_propagate_last_layer(data, labels, layer_outputs[i-1], self.loss_function, self.lr)
            else:
                delta = self.layers[i-1].back_propagate(delta, layer_outputs[i], self.layers[i].get_weights(), layer_outputs[i-1], self.lr)

    def save_weights(self, path: str, weights_file: str, bias_file: str):
        # save weights in file
        all_weights = []
        all_biases = []
        for layer in self.layers:
            all_weights.append(layer.get_weights())
            all_biases.append(layer.get_biases())
        np.save(path + "/" + weights_file, all_weights, allow_pickle=True)
        np.save(path + "/" + bias_file, all_biases, allow_pickle=True)

    def load_weights(self, path: str, weights_file: str, bias_file: str):
        # load weights from file
        all_weights = np.load(path + "/" + weights_file, allow_pickle=True)
        all_biases = np.load(path + "/" + bias_file, allow_pickle=True)
        for layer in range(len(self.layers)):
            self.layers[layer].set_weights(all_weights[layer])
            self.layers[layer].set_biases(all_biases[layer])


def main():
    nn = NeuralNetwork([2, 4, 1], [Sigmoid, Sigmoid], MSE, 0.1)
    data = np.array([
        [0, 1],
        [0, 0],
        [1, 1],
        [1, 0]
    ])
    labels = np.array([
        [1],
        [0],
        [0],
        [1]
    ])
    print(nn.predict(data))
    for i in range(20000):
        nn.train(data, labels)
        print(MSE.calculate(nn.predict(data), labels))
    print(nn.predict(data))


if __name__ == "__main__":
    main()

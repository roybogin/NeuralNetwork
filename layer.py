import abc
import numpy as np
from activation import Activation
from losses import Loss


class Layer(abc.ABC):
    @abc.abstractmethod
    def feed_forward(self):     # feed forward the data
        pass

    @abc.abstractmethod
    def back_propagate(self):   # back propagate a layer
        pass

    @abc.abstractmethod
    def back_propagate_last_layer(self):    # back propagate the last layer
        pass


class Dense(Layer):

    def __init__(self, input_nodes: int, output_nodes: int, activation: Activation, min_init_weight=-0.5, max_init_weight=0.5):
        self.weights = (max_init_weight - min_init_weight) * np.random.rand(input_nodes, output_nodes) + min_init_weight
        self.biases = (max_init_weight - min_init_weight) * np.random.rand(output_nodes) + min_init_weight
        self.activation = activation

    def feed_forward(self, data: np.ndarray):
        data = data.dot(self.weights) + self.biases
        data = self.activation.calculate(data)
        return data

    def back_propagate(self, delta: np.ndarray, next_layer_outputs: np.ndarray, next_layer_weights: np.ndarray, layer_outputs: np.ndarray, lr:float):
        delta = self.activation.derivative(next_layer_outputs) * np.dot(delta, next_layer_weights.T)
        lr = lr / delta.shape[0]    # divide by batch size
        self.weights -= lr * np.dot(layer_outputs.T, delta)
        self.biases -= lr * np.sum(delta, axis=0)
        return delta

    def back_propagate_last_layer(self, data: np.ndarray, labels: np.ndarray, layer_outputs:np.ndarray, loss_func: Loss, lr: float):
        delta = self.activation.derivative(data) * loss_func.derivative(data, labels)
        lr = lr / delta.shape[0]    # divide by batch size
        self.weights -= lr * np.dot(layer_outputs.T, delta)
        self.biases -= lr * np.sum(delta, axis=0)
        return delta

    # getters and setters
    def get_weights(self):
        return self.weights

    def get_biases(self):
        return self.biases

    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    def set_biases(self, biases: np.ndarray):
        self.biases = biases

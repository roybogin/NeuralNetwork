import abc
import numpy as np
from activation import Activation
from losses import Loss
from typing import List

class Layer(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def feed_forward():
        pass

    @staticmethod
    @abc.abstractmethod
    def back_propagate():
        pass

    @staticmethod
    @abc.abstractmethod
    def back_propagate_last_layer():
        pass


class Dense(Layer):
    @staticmethod
    def feed_forward(data: np.ndarray, weights: np.ndarray, biases: np.ndarray, activation: Activation):
        data = data.dot(weights) + biases
        data = activation.calculate(data)
        return data

    @staticmethod
    def back_propagate(delta: np.ndarray, next_layer_outputs: np.ndarray, next_layer_weights: np.ndarray, layer_weights: np.ndarray, biases: np.ndarray, layer_outputs:np.ndarray, activation: Activation, lr:float, vanish_grad=False):
        delta = activation.derivative(next_layer_outputs) * np.dot(delta, next_layer_weights.T)
        if not vanish_grad:
            layer_weights -= lr * np.dot(layer_outputs.T, delta)
            biases -= lr * np.sum(delta, axis=0)
        else:
            layer_weights -= lr * np.sign(np.dot(layer_outputs.T, delta))
            biases -= lr * np.sign(np.sum(delta, axis=0))
        return delta
    @staticmethod
    def back_propagate_last_layer(data: np.ndarray, labels: np.ndarray, layer_weights: np.ndarray, biases: np.ndarray, layer_outputs:np.ndarray, loss_func: Loss, activation: Activation, lr: float, vanish_grad=False):
        delta = activation.derivative(data) * loss_func.derivative(data, labels)
        if not vanish_grad:
            layer_weights -= lr * np.dot(layer_outputs.T, delta)
            biases -= lr * np.sum(delta, axis=0)
        else:
            layer_weights -= lr * np.sign(np.dot(layer_outputs.T, delta))
            biases -= lr * np.sign(np.sum(delta, axis=0))
        return delta

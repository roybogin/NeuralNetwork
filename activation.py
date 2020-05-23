import abc
import numpy as np


class Activation(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(x: np.ndarray):
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(x: np.ndarray):  # with respect to function output
        pass


class Sigmoid(Activation):
    @staticmethod
    def calculate(x: np.ndarray):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def derivative(x: np.ndarray):
        return x * (1-x)


class Tanh(Activation):
    @staticmethod
    def calculate(x: np.ndarray):
        return np.tanh(x)

    @staticmethod
    def derivative(x: np.ndarray):
        return 1 - x * x


class Relu(Activation):
    @staticmethod
    def calculate(x: np.ndarray):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x: np.ndarray):
        return 1 * (x > 0)


class Linear(Activation):
    @staticmethod
    def calculate(x: np.ndarray):
        return x

    @staticmethod
    def derivative(x: np.ndarray):
        return np.ones(x.shape)


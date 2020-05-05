import abc
import numpy as np
import warnings

warnings.simplefilter("error")

class Activation(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(x):
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(x):
        pass


class Sigmoid(Activation):
    @staticmethod
    def calculate(x):
        try:
            return 1/(1 + np.exp(-x))
        except:
            pass

    @staticmethod
    def derivative(x):
        return x * (1-x)


class Tanh(Activation):
    @staticmethod
    def calculate(x):
        return np.tanh(x)

    @staticmethod
    def derivative(x):
        return 1 - x * x


class Relu(Activation):
    @staticmethod
    def calculate(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x):
        return 1 * (x > 0)


class Linear(Activation):
    @staticmethod
    def calculate(x):
        return x

    @staticmethod
    def derivative(x):
        return np.ones(x.shape)


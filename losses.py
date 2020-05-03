import abc
import numpy as np
import warnings

#warnings.simplefilter('error')
class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        pass


class SSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return (labels - output) ** 2

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return output - labels

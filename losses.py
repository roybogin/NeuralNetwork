import abc
import numpy as np
import warnings

warnings.simplefilter('error')
class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        pass


class MSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        try:
            return (labels - output) ** 2
        except:
            pass

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        try:
            x = 2 * (labels - output)
        except:
            pass
        return x

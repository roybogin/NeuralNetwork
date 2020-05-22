import abc
import numpy as np


class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):  # (batch_size, 1)
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray):  # (batch_size, output_num)
        pass


class SSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        if len(labels.shape)==1:
            return np.sum(np.atleast_2d((labels - output)**2), axis=0)
        return np.sum((labels - output ** 2), axis=0)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2 * (output-labels)


class MSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        if len(labels.shape)==1:
            return np.mean(np.atleast_2d((labels - output)**2), axis=0)
        return np.mean((labels - output ** 2), axis=0)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2/len(labels) * (output-labels)


import abc
import numpy as np


class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):  # mean over (batch_size, 1)
        pass

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray):  # (batch_size, output_num)
        pass


class SSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return np.mean(np.sum((labels - output) ** 2, axis=1))

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2 * (output - labels)


class MSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return np.mean(np.mean((labels - output) ** 2, axis=1))

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2/labels.shape[1] * (output - labels)

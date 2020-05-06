import abc
import cupy as np


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
        return (labels - output) ** 2

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2 * (output-labels)


class MSE(Loss):
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return (labels - output) ** 2/len(labels)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2/len(labels) * (output-labels)


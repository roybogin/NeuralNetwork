import abc
import numpy as np


class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):  # calculate value of loss function
        pass    # mean over (batch size, 1)

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray): # calculate derivative of loss function
        pass    # return shape = (batch_size, output_num)


class SSE(Loss):    # Sum of Squared Errors
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        loss = np.sum((labels - output) ** 2, axis=1)
        return np.mean(loss)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2 * (output-labels)


class MSE(Loss):    # Mean of Squared Errors
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        loss = np.mean((labels - output) ** 2, axis=1)
        return np.mean(loss)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2/labels.shape[1] * (output - labels)


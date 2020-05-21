import abc
import numpy as np


class Loss(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def calculate(output: np.ndarray, labels: np.ndarray):  # calculate value of loss function
        pass    # return shape = (batch_size, 1)

    @staticmethod
    @abc.abstractmethod
    def derivative(output: np.ndarray, labels: np.ndarray): # calculate derivative of loss function
        pass    # return shape = (batch_size, output_num)


class SSE(Loss):    # Sum of Squared Errors
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return (labels - output) ** 2

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2 * (output-labels)


class MSE(Loss):    # Mean of Squared Errors
    @staticmethod
    def calculate(output: np.ndarray, labels: np.ndarray):
        return (labels - output) ** 2/len(labels)

    @staticmethod
    def derivative(output: np.ndarray, labels: np.ndarray):
        return 2/len(labels) * (output-labels)


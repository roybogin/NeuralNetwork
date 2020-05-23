import abc
import numpy as np


class NetworkModel(abc.ABC):

    @abc.abstractmethod
    def train(self, my_input: np.ndarray, labels: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, data: np.ndarray):
        pass

    @abc.abstractmethod
    def save_weights(self):
        pass

    @abc.abstractmethod
    def load_weights(self):
        pass

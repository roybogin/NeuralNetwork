import keras
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from network_model import NetworkModel
from keras.utils import to_categorical


import numpy as np


class CNN(NetworkModel):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape = (28, 28, 1)))
        self.model.add(Conv2D(32, kernel_size=3, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(10, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.05), metrics=['accuracy'])

    def predict(self, data: np.ndarray):
        return self.model.predict(data)

    def train(self, my_input: np.ndarray, labels: np.ndarray):
        return self.model.fit(my_input, labels, verbose=0, epochs=5, batch_size=128, validation_data=(my_input, labels))


def main():
    nn = CNN()
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 28, 28, 1)
    x_test = x_test.reshape(10000, 28, 28, 1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    print(nn.model.evaluate(x_test, y_test, verbose=0))
    for i in range(1):
        nn.train(x_train, y_train)
        print("did " + str(i))
    print(nn.model.evaluate(x_test, y_test, verbose=0))




if __name__ == "__main__":
    main()

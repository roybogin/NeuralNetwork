import abc
import socket
import numpy as np


class Env(abc.ABC):
    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def input_num(self):
        pass

    @abc.abstractmethod
    def action_num(self):
        pass

    @abc.abstractmethod
    def step(self, action):
        pass

    @abc.abstractmethod
    def close(self):
        pass


class MySock:
    def __init__(self, host, port):
        self.sock = socket.socket()
        self.sock.connect((host, port))

    def send(self, message):
        if len(message) > 0:
            self.sock.send(message.encode())

    def recv(self):
        return self.sock.recv(1024).decode()


class SocketEnv:
    def __init__(self, host, port):
        self.sock = MySock(host, port)

    def reset(self):
        self.sock.send("reset")
        return np.fromstring(self.sock.recv(), sep=", ", dtype=np.int16)

    def input_num(self):
        self.sock.send("input_num")
        return int(self.sock.recv())

    def input_shape(self):
        self.sock.send("input_shape")
        recv = self.sock.recv()
        arr = recv.split("#")
        return int(arr[0]), int(arr[1])

    def action_num(self):
        self.sock.send("action_num")
        return int(self.sock.recv())

    def step(self, action):
        self.sock.send("step " + str(action))
        recv = self.sock.recv()
        arr = recv.split("#")
        observation = np.fromstring(arr[0], sep=", ", dtype=np.int16)
        reward = int(arr[1])
        done = arr[2] == "true"
        did_win = arr[3] == "true"
        return observation, reward, done, did_win

    def close(self):
        self.sock.send("close")
        self.sock.sock.close()

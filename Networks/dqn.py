from network_model import NetworkModel
import numpy as np
from typing import List, Callable
from losses import Loss
from statistics import mean


class DQN:
    def __init__(self, model: NetworkModel, num_actions: int, loss_func: Loss, gamma: float, max_experiences: int,
                 min_experiences: int, batch_size: int):
        self.model = model
        self.num_actions = num_actions  # output number
        self.batch_size = batch_size    # number of experiences to train by at one training session
        self.loss_function = loss_func
        self.gamma = gamma  # reward decay for bellman equation
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}     # experience buffer
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs: np.ndarray):
        return self.model.predict(np.atleast_2d(inputs.astype(float)))

    def train(self, target_net):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        # choose random experiences
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids], dtype=np.int16)
        actions = np.asarray([self.experience['a'][i] for i in ids], dtype=np.int16)
        rewards = np.asarray([self.experience['r'][i] for i in ids], dtype=np.int16)
        states_next = np.asarray([self.experience['s2'][i] for i in ids], dtype=np.int16)
        dones = np.asarray([self.experience['done'][i] for i in ids], dtype=np.int16)
        # predict by target network
        value_next = np.max(target_net.predict(states_next), axis=1)
        # actual values by bellman equation
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
        prediction = self.predict(states)
        prediction_list = []
        for i in range(self.batch_size):
            prediction_list.append(prediction[i][actions[i]])
            prediction[i][actions[i]] = actual_values[i]
        # train training network
        self.model.train(states, prediction)
        loss = self.loss_function.calculate(np.atleast_2d(np.array(prediction_list)).T, np.atleast_2d(np.array(actual_values)).T)
        # return mean of losses for batch
        return loss

    def get_legal_action(self, states: np.ndarray, epsilon: float, is_legal_move: Callable):    # get a legal action
        if np.random.random() < epsilon:
            # random if number is below epsilon
            legal_moves = np.array(list(filter(lambda x: is_legal_move(states, x), range(self.num_actions))))
            return np.random.choice(legal_moves)
        else:
            prediction = self.predict(np.atleast_2d(states))
            action = np.argmax(prediction)
            while not is_legal_move(states, action):
                # if action is not legal choose another
                prediction[0][action] = -float("Inf")
                action = np.argmax(prediction)
            return action

    def get_action(self, states: np.ndarray, epsilon: float):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            prediction = self.predict(np.atleast_2d(states))
            return np.argmax(prediction)[0]

    def add_experience(self, exp: dict):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)


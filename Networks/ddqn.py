from Networks.neural_network import NeuralNetwork
import cupy as np
from typing import List, Callable
from activation import Activation
from losses import Loss
from statistics import mean
from env import Env


class DDQN:
    def __init__(self, layer_nums: List[int], activations: List[Activation], loss_func: Loss, gamma: float,
                 max_experiences: int, min_experiences: int, batch_size: int, lr: float, min_init_weight=-0.5, max_init_weight=0.5):
        self.model = NeuralNetwork(layer_nums, activations, loss_func, lr, min_init_weight, max_init_weight)
        self.num_actions = layer_nums[-1]
        self.batch_size = batch_size
        self.loss_function = loss_func
        self.gamma = gamma
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs: np.ndarray):
        return self.model.predict(np.atleast_2d(inputs.astype(float)))

    def train(self, target_net):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(target_net.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
        prediction = self.predict(states)
        prediction_list = []
        for i in range(self.batch_size):
            prediction_list.append(prediction[i][actions[i]])
            prediction[i][actions[i]] = actual_values[i]
        self.model.train(states, prediction)
        loss = self.loss_function.calculate(np.array(prediction_list), np.array(actual_values))
        return mean(loss)

    def get_legal_action(self, states: np.ndarray, epsilon: float, is_legal_move: Callable):
        if np.random.random() < epsilon:
            legal_moves = np.array(list(filter(lambda x: is_legal_move(states, x), range(self.num_actions))))
            return np.random.choice(legal_moves)
        else:
            prediction = self.predict(np.atleast_2d(states))
            action = np.argmax(prediction)
            while not is_legal_move(states, action):
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

    def is_experience_in(self, state: np.ndarray, action: int):
        for i, act in enumerate(self.experience['a']):
            if act == action and all(state == self.experience['s'][i]):
                return True
        return False

    def copy_weights(self, train_net):
        self.model.layer_weights = np.copy(train_net.model.layer_weights)
        self.model.biases = np.copy(train_net.model.biases)


def play_game(env: Env, train_net: DDQN, target_net: DDQN, epsilon: float, copy_step: int, wins: int, is_legal_move=None, info=lambda: None):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        if is_legal_move is None:
            action = train_net.get_action(observations, epsilon)
        else:
            action = train_net.get_legal_action(observations, epsilon, is_legal_move)
        prev_observations = observations
        observations, reward, done, did_win = env.step(action)
        if did_win:
            wins += 1
        rewards += reward
        if done:
            env.reset()
        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        train_net.add_experience(exp)
        loss = train_net.train(target_net)
        losses.append(loss)
        iter += 1
        if iter % copy_step == 0:
            target_net.copy_weights(train_net)
    return rewards, mean(losses), wins, observations

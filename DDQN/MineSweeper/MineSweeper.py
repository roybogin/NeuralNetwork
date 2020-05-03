from Networks.neural_network import NeuralNetwork
from typing import List
from losses import SSE
from activation import Activation, Relu, Linear
from env import SocketEnv
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import copy

class DDQN:
    def __init__(self, num_states: int, num_actions: int, hidden_units: List[int], activations: List[Activation], gamma: float, max_experiences: int, min_experiences: int, batch_size: int, lr=0.01):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.model = NeuralNetwork([num_states] + hidden_units + [num_actions], activations, lr)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model.predict(np.atleast_2d(inputs.astype(float)))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(low=0, high=len(self.experience['s']), size=self.batch_size)
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(dones, rewards, rewards+self.gamma*value_next)
        prediction = self.predict(states)
        prediction_list = []
        for i in range(self.batch_size):
            prediction_list.append(prediction[i][actions[i]])
            prediction[i][actions[i]] = actual_values[i]
        self.model.train_ddqn(states, prediction)
        loss = SSE.calculate(np.array(prediction_list), np.array(actual_values))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        self.model.layer_weights = np.copy(TrainNet.model.layer_weights)
        self.model.biases = np.copy(TrainNet.model.biases)


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    global wins
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        if done and reward == 1:
            wins += 1
        rewards += reward
        if done:
            env.reset()

        exp = {'s': prev_observations, 'a': action, 'r': reward, 's2': observations, 'done': done}
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, np.ndarray):
            losses.append(np.mean(loss))
        else:
            losses.append(loss)

        iter += 1
        if iter % copy_step == 0 or (done and reward == 1):
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)


wins = 0
def main():
    global wins
    host = "127.0.0.1"
    port = 2000
    env = SocketEnv(host, port)
    gamma = 0
    copy_step = 10
    num_states = env.input_num()
    num_actions = env.action_num()
    hidden_units = [64] #todo: maybe change layers
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 0.01

    TrainNet = DDQN(num_states, num_actions, hidden_units, [Relu, Linear], gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = copy.deepcopy(TrainNet)

    #TargetNet.model.load_weights("saved_data/cp.ckpt")
    #TrainNet.model.load_weights("saved_data/cp.ckpt")

    episode_list = []
    avg_rwd_list = []
    losses_list = []
    win_list = []

    N = 100000
    total_rewards = np.empty(N)
    with open("saved_data/epsilon.txt", "r") as f:
        data = f.read()
        epsilon = 1#float(data)
    decay = 0.9999
    min_epsilon = 0.05
    try:
        for n in range(N):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, losses = play_game(env, TrainNet, TargetNet, epsilon, copy_step)
            total_rewards[n] = total_reward
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
            if n % 100 == 0:
                print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                      "episode loss:", losses, "episode wins:", wins)
                win_list.append(wins)
                episode_list.append(n)
                avg_rwd_list.append(avg_rewards)
                losses_list.append(losses)
                wins = 0
                #TargetNet.model.save_weights("saved_data/cp.ckpt")
                with open("saved_data/epsilon.txt", "w") as f:
                    f.write(str(epsilon))
        env.close()

    except:
        pass


    print("avg reward for last 100 episodes:", avg_rewards)

    _, (ax1, ax2, ax3) = plt.subplots(3)
    ax1.plot(episode_list, avg_rwd_list)
    ax1.set(xlabel="episode", ylabel="avg reward last 100")

    ax2.plot(episode_list, losses_list)
    ax2.set(xlabel="episode", ylabel="avg loss last 100")

    ax3.plot(episode_list, win_list)
    ax3.set(xlabel="episode", ylabel="wins last 100")
    plt.show()
    #TargetNet.model.save_weights("saved_data/cp.ckpt")

    with open("saved_data/epsilon.txt", "w") as f:
        f.write(str(epsilon))

    with open("saved_data/rewards.txt", "w") as f:
        f.write(str(avg_rwd_list))

    with open("saved_data/losses.txt", "w") as f:
        f.write(str(losses_list))

    plt.show()


if __name__ == '__main__':
    main()



from Networks.ddqn import DDQN
from Networks.neural_network import NeuralNetwork
from losses import SSE, MSE
from activation import Linear, Tanh
from env import SocketEnv, Env
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime
from statistics import mean
import pickle as pkl
import os


def main():
    host = "127.0.0.1"
    port = 2000
    env = SocketEnv(host, port)
    wins = 0    # how many wins in last calculation step
    folder_name = "ddqn"
    saving_path = "saved_data/" + folder_name
    dir_num = len([d for d in os.listdir(saving_path) if os.path.isdir(saving_path + "/" + d)])  # number of existing directories (to add another)
    take_from = saving_path + "/" + str(dir_num-1)  # directory to take data from
    os.mkdir(saving_path + "/" + str(dir_num))
    saving_path = saving_path + "/" + str(dir_num)  # directory to save data in
    gamma = 0.01    # DQN gamma
    copy_step = 10  # step to copy weights between networks
    loss_function = SSE
    layers = [env.input_num(), 64, 64, env.action_num()]
    max_experiences = 20000
    min_experiences = 1000
    decay = 0.9999  # epsilon decay
    min_epsilon = 0.01
    batch_size = 700
    lr = 5e-4
    calculation_step = 1000  # step for calculating the data for the plot
    monitoring_step = 30    # step to show info on console
    runs_number = int(7e5)  # how many runs to do
    train_from_start = False    # training from start or file
    estimate_time = True    # estimate time to end or show monitoring

    train_net = DDQN(NeuralNetwork(layers, [Tanh, Tanh, Linear], loss_function, lr), layers[-1], loss_function, gamma, max_experiences, min_experiences, batch_size)

    if dir_num == 0:    # if there are no data directories you have to train from start
        train_from_start = True

    if not train_from_start:
        train_net.model.load_weights(take_from, "weights.npy", "biases.npy")
    target_net = copy.deepcopy(train_net)

    episode_list = []
    avg_rwd_list = []
    losses_list = []
    win_list = []
    reveal_list = []

    total_rewards = np.empty(runs_number)
    total_losses = np.empty(runs_number)
    total_reveal_percent = np.empty(runs_number)
    if train_from_start:
        epsilon = 1
    else:
        with open(take_from + "/epsilon.txt", "r") as f:
            data = f.read()
            epsilon = float(data)
    try:
        if estimate_time:
            start_time = datetime.datetime.now()
        for n in range(runs_number):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, ep_losses, wins, end_board = play_game(env, train_net, target_net, epsilon, copy_step, wins) # play a game
            total_rewards[n] = total_reward
            total_losses[n] = ep_losses
            total_reveal_percent[n] = board_percent_revealed(end_board)
            if n % calculation_step == 0:
                # calculate data for plot
                avg_rewards = total_rewards[max(0, n - calculation_step):(n + 1)].mean()
                avg_losses = total_losses[max(0, n - calculation_step):(n + 1)].mean()
                avg_reveal_percents = total_reveal_percent[max(0, n - calculation_step):(n + 1)].mean()
                win_list.append(wins)
                reveal_list.append(avg_reveal_percents)
                episode_list.append(n)
                avg_rwd_list.append(avg_rewards)
                losses_list.append(avg_losses)
                wins = 0    # reset wins
                target_net.model.save_weights(saving_path, "weights.npy", "biases.npy")
                with open(saving_path + "/epsilon.txt", "w") as f:
                    f.write(str(epsilon))
            if n % monitoring_step == 0:
                if not estimate_time:
                    # show monitoring
                    avg_rewards = total_rewards[max(0, n - calculation_step):(n + 1)].mean()
                    avg_losses = total_losses[max(0, n - calculation_step):(n + 1)].mean()
                    avg_reveal_percents = total_reveal_percent[max(0, n - calculation_step):(n + 1)].mean()
                    print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last",
                          str(calculation_step) + "):", avg_rewards,
                          "avg loss (last", str(calculation_step) + "):", avg_losses, "last", calculation_step,
                          "episodes wins: ", wins, "avg revealed percent:", avg_reveal_percents)
                else:
                    print("plays percentage: ", str(100 * n / runs_number) + "%   estimated time: ",
                          str(estimate_runtime(n / runs_number, start_time)))

        env.close()

    except:  # if code is stopped at the middle, still calculate data and show plot
        pass

    finally:
        # calculate end of data
        avg_rewards = total_rewards[max(0, n - calculation_step):(n + 1)].mean()
        avg_losses = total_losses[max(0, n - calculation_step):(n + 1)].mean()
        avg_reveal_percents = total_reveal_percent[max(0, n - calculation_step):(n + 1)].mean()
        win_list.append(wins)
        reveal_list.append(avg_reveal_percents)
        episode_list.append(n)
        avg_rwd_list.append(avg_rewards)
        losses_list.append(avg_losses)

        print("avg reward for last", calculation_step, "episodes:", avg_rewards)
        print("total win percent", (np.sum(np.array(win_list)))/n)

        # plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.plot(episode_list, avg_rwd_list)
        ax1.set(xlabel="episode", ylabel="avg reward last " + str(calculation_step))

        ax2.plot(episode_list, losses_list)
        ax2.set(xlabel="episode", ylabel="avg loss last " + str(calculation_step))

        ax3.plot(episode_list, win_list)
        ax3.set(xlabel="episode", ylabel="wins last " + str(calculation_step))

        ax4.plot(episode_list, reveal_list)
        ax4.set(xlabel="episode", ylabel="avg percent board last " + str(calculation_step))

        target_net.model.save_weights(saving_path, "weights.npy", "biases.npy")

        with open(saving_path + "/epsilon.txt", "w") as f:
            f.write(str(epsilon))

        with open(saving_path + "/rewards.txt", "w") as f:
            f.write(str(avg_rwd_list))

        with open(saving_path + "/losses.txt", "w") as f:
            f.write(str(losses_list))

        with open(saving_path + "/wins.txt", "w") as f:
            f.write(str(win_list))

        with open(saving_path + "/reveals.txt", "w") as f:
            f.write(str(reveal_list))

        with open(saving_path + "/info.txt", "w") as f:
            f.write("lr: " + str(lr) + "    runs: " + str(n))

        pkl.dump(fig, open(saving_path + "/plot.p", 'wb'))
        plt.show()


def play_game(env: Env, train_net: DDQN, target_net: DDQN, epsilon: float, copy_step: int, wins: int):
    rewards = 0
    iter = 0
    done = False
    observation = env.reset()
    losses = list()
    while not done:
        # select legal move
        action = train_net.get_legal_action(observation, epsilon, is_legal_move)
        prev_observation = observation
        observation, reward, done, did_win = env.step(action)
        if did_win:
            wins += 1
        rewards += reward
        if done:
            env.reset()
        exp = {'s': prev_observation, 'a': action, 'r': reward, 's2': observation, 'done': done}
        # add experience for replay
        train_net.add_experience(exp)
        loss = train_net.train(target_net)
        losses.append(loss)
        iter += 1
        if iter % copy_step == 0:
            # copy weights between networks
            target_net.model.layers = copy.deepcopy(train_net.model.layers)
    return rewards, mean(losses), wins, observation


def board_percent_revealed(board: np.ndarray):
    # calculate percent of board that is revealed
    num_revealed = np.sum(((board >= 0) * (board <= 8)))
    return num_revealed/board.size


def is_legal_move(state: np.ndarray, action: int):
    # a move is legal if the panel isn't shown (value of -5)
    return state[action] == -5


def estimate_runtime(percent_done: float, starting_time: datetime.datetime):
    if percent_done == 0:
        return 0
    now = datetime.datetime.now()
    diff = now - starting_time
    diff = diff * ((1 - percent_done) / percent_done)
    return diff - datetime.timedelta(microseconds=diff.microseconds)


if __name__ == '__main__':
    main()



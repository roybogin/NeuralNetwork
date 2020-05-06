from Networks.ddqn import DDQN, play_game
from losses import MSE
from activation import Relu, Linear, Sigmoid, Tanh
from env import SocketEnv
import numpy as np
import matplotlib.pyplot as plt
import copy
import datetime


wins = 0


def main():
    global wins
    host = "127.0.0.1"
    port = 2000
    env = SocketEnv(host, port)
    saving_path = "saved_data"
    weights_file = "weights_4by4_sig_gam001.npy"  # "weights.npy"
    bias_file = "weights_4by4_sig_gam001.npy"  # "biases.npy"
    gamma = 0.01
    copy_step = 10
    layers = [env.input_num(), 64, env.action_num()]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 0.05
    calculation_step = 400
    runs_number = int(2e6)
    train_from_start = True
    estimate_time = False

    train_net = DDQN(layers, [Sigmoid, Linear], MSE, gamma, max_experiences, min_experiences, batch_size, lr)
    if not train_from_start:
        train_net.model.load_weights(saving_path, weights_file, bias_file)
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
        with open(saving_path + "/epsilon.txt", "r") as f:
            data = f.read()
            epsilon = float(data)
    decay = 0.9999
    min_epsilon = 0.05
    try:
        if estimate_time:
            start_time = datetime.datetime.now()
        for n in range(runs_number):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, ep_losses, wins, end_board = play_game(env, train_net, target_net, epsilon, copy_step, wins, is_legal_move=is_legal_move)
            if estimate_time:
                print("plays percentage: ", str(100 * n/runs_number) + "%   estimated time: ", str(estimate_runtime(n/runs_number, start_time)))
            total_rewards[n] = total_reward
            total_losses[n] = ep_losses
            total_reveal_percent[n] = board_percent_revealed(end_board)
            if n % calculation_step == 0:
                avg_rewards = total_rewards[max(0, n - calculation_step):(n + 1)].mean()
                avg_losses = total_losses[max(0, n - calculation_step):(n + 1)].mean()
                avg_reveal_percents = total_reveal_percent[max(0, n - calculation_step):(n + 1)].mean()
                if not estimate_time:
                    print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last",
                          str(calculation_step) + "):", avg_rewards,
                          "avg loss (last", str(calculation_step) + "):", avg_losses, "last", calculation_step,
                          "episodes wins: ", wins, "avg revealed percent:", avg_reveal_percents)
                win_list.append(wins)
                reveal_list.append(avg_reveal_percents)
                episode_list.append(n)
                avg_rwd_list.append(avg_rewards)
                losses_list.append(avg_losses)
                wins = 0
                target_net.model.save_weights(saving_path, weights_file, bias_file)
                with open(saving_path + "/epsilon.txt", "w") as f:
                    f.write(str(epsilon))
        env.close()

    except ConnectionError:
        pass


    print("avg reward for last", calculation_step, "episodes:", avg_rewards)
    print("total win percent", (np.sum(np.array(win_list)))/runs_number)

    _, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.plot(episode_list, avg_rwd_list)
    ax1.set(xlabel="episode", ylabel="avg reward last " + str(calculation_step))

    ax2.plot(episode_list, losses_list)
    ax2.set(xlabel="episode", ylabel="avg loss last" + str(calculation_step))

    ax3.plot(episode_list, win_list)
    ax3.set(xlabel="episode", ylabel="wins last" + str(calculation_step))

    ax4.plot(episode_list, reveal_list)
    ax4.set(xlabel="episode", ylabel="avg percent board last" + str(calculation_step))

    target_net.model.save_weights(saving_path, weights_file, bias_file)

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

    plt.show()


def board_percent_revealed(board: np.ndarray):
    num_revealed = np.sum(((board >= 0) * (board <= 8)))
    return num_revealed/board.size


def is_legal_move(state: np.ndarray, action: int):
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



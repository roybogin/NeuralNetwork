from Networks.ddqn import DDQN, play_game
from losses import MSE
from activation import Relu, Linear, Sigmoid
from env import SocketEnv
import numpy as np
import matplotlib.pyplot as plt
import copy


wins = 0


def main():
    global wins
    host = "127.0.0.1"
    port = 2000
    env = SocketEnv(host, port)
    saving_path = "saved_data"
    weights_file = "weights.npy"
    bias_file = "biases.npy"
    gamma = 0
    copy_step = 10
    layers = [env.input_num(), 64, env.action_num()]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 32
    lr = 0.1
    runs_number = 200
    train_from_start = False

    train_net = DDQN(layers, [Relu, Linear], MSE, gamma, max_experiences, min_experiences, batch_size, lr)
    if not train_from_start:
        train_net.model.load_weights(saving_path, weights_file, bias_file)
    target_net = copy.deepcopy(train_net)

    episode_list = []
    avg_rwd_list = []
    losses_list = []
    win_list = []

    total_rewards = np.empty(runs_number)
    total_losses = np.empty(runs_number)
    if train_from_start:
        epsilon = 1
    else:
        with open(saving_path + "/epsilon.txt", "r") as f:
            data = f.read()
            epsilon = float(data)
    decay = 0.9999
    min_epsilon = 0.05
    try:
        for n in range(runs_number):
            epsilon = max(min_epsilon, epsilon * decay)
            total_reward, ep_losses, wins = play_game(env, train_net, target_net, epsilon, copy_step, wins)
            total_rewards[n] = total_reward
            total_losses[n] = ep_losses
            avg_rewards = total_rewards[max(0, n - 100):(n + 1)].mean()
            avg_losses = total_losses[max(0, n - 100):(n + 1)].mean()
            if n % 100 == 0:
                print("episode:", n, "episode reward:", total_reward, "eps:", epsilon, "avg reward (last 100):", avg_rewards,
                      "avg loss (last 100):", avg_losses, "last 100 episode wins:", wins)
                win_list.append(wins)
                episode_list.append(n)
                avg_rwd_list.append(avg_rewards)
                losses_list.append(avg_losses)
                wins = 0
                target_net.model.save_weights(saving_path, weights_file, bias_file)
                with open(saving_path + "/epsilon.txt", "w") as f:
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

    target_net.model.save_weights(saving_path, weights_file, bias_file)

    with open(saving_path + "/epsilon.txt", "w") as f:
        f.write(str(epsilon))

    with open(saving_path + "/rewards.txt", "w") as f:
        f.write(str(avg_rwd_list))

    with open(saving_path + "/losses.txt", "w") as f:
        f.write(str(losses_list))

    plt.show()


if __name__ == '__main__':
    main()



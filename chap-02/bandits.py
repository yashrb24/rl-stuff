import matplotlib.pyplot as plt
from tqdm import trange
from base import *


def simulate_epsilon_greedy_stationary(runs=2000, steps=1000, initial=0, k_arms=10):
    arm_means = np.random.normal(loc=0, scale=1, size=k_arms)
    epsilons = [0, 0.01, 0.1, 0.5]
    best_arm = np.argmax(arm_means)

    rewards = np.zeros((len(epsilons), runs, steps))
    optimal = np.zeros((len(epsilons), runs, steps))

    for i, epsilon in enumerate(epsilons):
        bandits = [
            Bandit(k_arms=k_arms, epsilon=epsilon, initial=initial) for _ in range(runs)
        ]
        for run in trange(runs):
            bandit = bandits[run]
            for step in range(steps):
                action = bandit.action()
                reward = np.random.normal(loc=arm_means[action], scale=1)
                bandit.update(action, reward)

                rewards[i][run][step] = reward
                optimal[i][run][step] = action == best_arm

    steps_array = np.arange(1, steps + 1)

    # plot for average reward
    plt.subplot(2, 1, 1)
    for i in range(len(epsilons)):
        plt.plot(np.mean(rewards[i], axis=0), label=f"epsilon = {epsilons[i]}")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    # plot for percentage of times optimal action was taken
    plt.subplot(2, 1, 2)
    for i in range(len(epsilons)):
        plt.plot(
            np.mean(np.cumsum(optimal[i], axis=1), axis=0) * 100 / steps_array ,
            label=f"epsilon = {epsilons[i]}",
        )
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")

    plt.legend()
    plt.show()


def simulate_epsilon_greedy_non_stationary(runs=500, steps=10000, k_arms=10):
    arm_means = np.zeros(k_arms)

    bandits = [
        [Bandit(average_estimate=True) for _ in range(runs)],
        [Bandit(average_estimate=False) for _ in range(runs)],
    ]

    random_walks = np.random.normal(loc=0, scale=0.01, size=(runs, steps, k_arms))

    rewards = np.zeros((2, runs, steps))
    optimal = np.zeros((2, runs, steps))

    for run in trange(runs):
        for step in range(steps):
            best_arm = np.argmax(arm_means)

            for i in range(2):
                bandit = bandits[i][run]
                action = bandit.action()
                reward = np.random.normal(loc=arm_means[action], scale=1)
                bandit.update(action, reward)

                rewards[i][run][step] = reward
                optimal[i][run][step] = (action == best_arm)

            arm_means += random_walks[run][step]

    steps_array = np.arange(1, steps + 1)

    # plot for average reward
    plt.subplot(2, 1, 1)
    plt.plot(np.mean(rewards[0], axis=0), label="with constant step size")
    plt.plot(np.mean(rewards[1], axis=0), label="with sample average")
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    # plot for percentage of times optimal action was taken
    plt.subplot(2, 1, 2)
    plt.plot(
        np.mean(np.cumsum(optimal[0], axis=1), axis=0) / steps_array * 100,
        label="with constant step size",
    )
    plt.plot(
        np.mean(np.cumsum(optimal[1], axis=1), axis=0) / steps_array * 100,
        label="with sample average",
    )
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    plt.show()



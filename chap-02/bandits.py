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

    # plots
    plot_rewards_optimal_actions(
        rewards,
        optimal,
        reward_labels=[f"epsilon = {eps}" for eps in epsilons],
        optimal_labels=[f"epsilon = {eps}" for eps in epsilons],
    )


def simulate_epsilon_greedy_non_stationary(runs=500, steps=10000, k_arms=10):
    arm_means = np.zeros(k_arms)

    bandits = [
        [Bandit(use_average=True) for _ in range(runs)],
        [Bandit(use_average=False) for _ in range(runs)],
    ]

    random_walks = np.random.normal(loc=0, scale=0.01, size=(runs, steps, k_arms))

    rewards = np.zeros((2, runs, steps))
    optimal = np.zeros((2, runs, steps))

    for run in trange(runs):
        tmp_arm_means = arm_means.copy()

        for step in range(steps):
            best_arm = np.argmax(tmp_arm_means)

            for i in range(2):
                bandit = bandits[i][run]
                action = bandit.action()
                reward = np.random.normal(loc=tmp_arm_means[action], scale=1)
                bandit.update(action, reward)

                rewards[i][run][step] = reward
                optimal[i][run][step] = action == best_arm

            tmp_arm_means += random_walks[run][step]

    # plots
    plot_rewards_optimal_actions(
        rewards,
        optimal,
        reward_labels=["average step size", "constant step size"],
        optimal_labels=["average step size", "constant step size"],
    )


def simulate_optimistic_initial(
    runs=2000, steps=1000, initial=5, k_arms=10, epsilon=0.1
):
    arm_means = np.random.normal(loc=0, scale=1, size=k_arms)
    best_arm = np.argmax(arm_means)

    bandits = [
        [
            Bandit(epsilon=0, use_average=False, initial=initial)
            for _ in range(runs)
        ],
        [
            Bandit(epsilon=epsilon, use_average=False, initial=0)
            for _ in range(runs)
        ],
    ]

    rewards = np.zeros((2, runs, steps))
    optimal = np.zeros((2, runs, steps))

    for run in trange(runs):
        for step in range(steps):
            for i in range(2):
                bandit = bandits[i][run]
                action = bandit.action()
                reward = np.random.normal(loc=arm_means[action], scale=1)
                bandit.update(action=action, reward=reward)

                rewards[i][run][step] = reward
                optimal[i][run][step] = best_arm == action

    # plots
    plot_rewards_optimal_actions(
        rewards=rewards,
        optimal=optimal,
        reward_labels=["Optimisitc, Greedy", "Realistic, Epsilon-Greedy"],
        optimal_labels=["Optimisitc, Greedy", "Realistic, Epsilon-Greedy"],
    )


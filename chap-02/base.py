import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(
        self,
        k_arms=10,
        epsilon=0.1,
        initial=0,
        use_average=True,
        step_size=0.1,
        use_ucb=False,
        ucb_constant=1,
    ):
        self.timestep = 0
        self.k_arms = k_arms
        self.q_values = np.ones(k_arms) * initial
        self.n_values = np.zeros(k_arms)
        self.epsilon = epsilon
        self.use_average = use_average
        self.step_size = step_size
        self.use_ucb = use_ucb
        self.ucb_constant = ucb_constant

    def action(self):
        action = None
        if self.use_ucb == True:
            for i, n in enumerate(self.n_values):
                if n == 0:
                    action = i
                    break
            else:
                upper_bound_array = self.q_values + self.ucb_constant * np.sqrt(np.log(self.timestep) / self.n_values)
                action = np.argmax(upper_bound_array)

        elif np.random.random() < self.epsilon:
            action = np.random.randint(0, self.k_arms)

        else:
            action = np.argmax(self.q_values)

        self.timestep += 1
        return action

    def update(self, action, reward):
        self.n_values[action] += 1
        if self.use_average == True:
            self.q_values[action] += (1 / self.n_values[action]) * (
                reward - self.q_values[action]
            )
        else:
            self.q_values[action] += self.step_size * (reward - self.q_values[action])


def plot_rewards_optimal_actions(rewards, optimal, reward_labels, optimal_labels):
    cnt = rewards.shape[0]
    steps_array = np.arange(1, rewards.shape[-1] + 1)

    # plot for average reward
    plt.subplot(2, 1, 1)
    for i in range(cnt):
        plt.plot(np.mean(rewards[i], axis=0), label=reward_labels[i])
    plt.xlabel("Steps")
    plt.ylabel("Average Reward")
    plt.legend()

    # plot for percentage of times optimal action was taken
    plt.subplot(2, 1, 2)
    for i in range(cnt):
        plt.plot(
            np.mean(np.cumsum(optimal[i], axis=1), axis=0) * 100 / steps_array,
            label=optimal_labels[i],
        )
    plt.xlabel("Steps")
    plt.ylabel("% Optimal Action")
    plt.legend()

    plt.show()

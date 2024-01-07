import numpy as np

class Bandit:
    def __init__(
        self, k_arms=5, epsilon=0.1, initial=0, average_estimate=True, step_size=1e-5
    ):
        self.k_arms = k_arms
        self.q_values = np.ones(k_arms) * initial
        self.n_values = np.zeros(k_arms)
        self.epsilon = epsilon
        self.average_estimate = average_estimate
        self.step_size = step_size

    def action(self):
        action = None
        if np.random.random() < self.epsilon:
            action = np.random.randint(0, self.k_arms)
        else:
            action = np.argmax(self.q_values)

        return action

    def update(self, action, reward):
        self.n_values[action] += 1
        if self.average_estimate:
            self.q_values[action] += (1 / self.n_values[action]) * (reward - self.q_values[action])
        else:
            self.q_values[action] += self.step_size * (reward - self.q_values[action])

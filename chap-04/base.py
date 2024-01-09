import numpy as np


class GridWorld:
    def __init__(
        self,
        rows=4,
        columns=4,
        reward_per_transition=-1,
        variable_movement=False,
        variable_movement_probability=0,
        terminal_states=None,
    ):
        self.rows = rows
        self.columns = columns
        self.reward_per_transition = reward_per_transition
        self.variable_movement = variable_movement
        self.variable_movement_probability = variable_movement_probability
        self.terminal_states = (
            set({(rows - 1, columns - 1), (0, 0)})
            if terminal_states is None
            else terminal_states
        )

        self.movements = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1),
        }
        self.actions = ["up", "down", "left", "right"]

    def take_action(self, state, action):
        if self.variable_movement == True:
            if np.random.random() < self.variable_movement_probability:
                action = np.random.choice(self.actions)

        dx, dy = self.movements[action]
        x, y = state
        new_x, new_y = x + dx, y + dy

        if new_x < 0 or new_x >= self.rows or new_y < 0 or new_y >= self.columns:
            new_x, new_y = x, y

        new_state = (new_x, new_y)
        reward = self.reward_per_transition

        return new_state, reward


class Agent:
    def __init__(
        self,
        environment,
        discount_factor=1,
        policy_evaluation_steps=1e6,
        threshold=1e-4,
    ):
        self.environment = environment
        self.discount_factor = discount_factor
        self.policy_evaluation_steps = policy_evaluation_steps
        self.threshold = threshold

        self.state_values = np.zeros((self.environment.rows, self.environment.columns))
        self.policy = np.full(
            (
                self.environment.rows,
                self.environment.columns,
                len(self.environment.actions),
            ),
            fill_value=0.25,
        )

    def evalute_policy(self, inplace=False):
        for _ in range(int(self.policy_evaluation_steps)):
            new_state_values = np.zeros_like(self.state_values)

            for row in range(self.environment.rows):
                for column in range(self.environment.columns):
                    
                    state = (row, column)
                    if state in self.environment.terminal_states:
                        continue

                    action_probabilities = self.policy[row, column]

                    for action_index, action_probability in enumerate(
                        action_probabilities
                    ):
                        action = self.environment.actions[action_index]
                        new_state, reward = self.environment.take_action(state, action)
                        new_state_values[row, column] += action_probability * (
                            reward + self.discount_factor * self.state_values[new_state]
                        )
            new_state_values = np.ceil(new_state_values * 10) / 10
            
            if np.max(np.abs(new_state_values - self.state_values)) < self.threshold:
                break

            self.state_values = new_state_values

    def improve_policy(self):
        for row in range(self.environment.rows):
            for column in range(self.environment.columns):
                state = (row, column)
                action_values = []

                for action in self.environment.actions:
                    new_state, reward = self.environment.take_action(state, action)
                    action_values.append(
                        reward + self.discount_factor * self.state_values[new_state]
                    )

                best_actions = np.max(action_values)
                best_actions_count = np.sum(action_values == best_actions)
                action_probability = 1 / best_actions_count

                self.policy[row, column] = np.where(
                    action_values == best_actions, action_probability, 0
                )

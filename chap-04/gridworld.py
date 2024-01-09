from base import *

env = GridWorld(rows=4, columns=4)
agent = Agent(env, discount_factor=1)

agent.evalute_policy()
print("State values after policy evaluation:")
print(agent.state_values)

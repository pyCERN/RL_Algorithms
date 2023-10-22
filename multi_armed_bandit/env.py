import numpy as np

class Environment:
    def __init__(self):
        self.action_size = 10
        self.true_expected_rewards = np.random.uniform(-3, 3, self.action_size) # q*(a)
        self.action_values = [0 for _ in range(self.action_size)]
        self.num_action_selections = [0 for _ in range(self.action_size)]
        self.max_steps = 5

    def evaluate(self, reward, action):
        self.num_action_selections[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.num_action_selections[action]

    def step(self, action):
        epsilon = np.random.uniform()
        if epsilon < 0.5:
            next_action = np.argmax(self.action_values[action])
        else:
            next_action = np.random.choice(self.action_size)
        reward = np.random.normal(self.true_expected_rewards[next_action], 1)
        return next_action, reward
    
    def run(self):
        action = np.random.choice(self.action_size)
        for step in range(self.max_steps):
            print(action)
            action, reward = self.step(action)
            self.evaluate(reward, action)
            print(self.action_values)

# class Agent:
#     def __init__(self):
#         self.trajectory = [[0] for _ in range(10)]
#         self.action_values = [[] for _ in range(10)]
#         # self.action_value = 0
#         self.environment = Environment()


# env = Environment()
# agent = Agent()
# agent.step()
env = Environment()
env.run()
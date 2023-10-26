import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# from ..Plotter.Plotter import DataCollector

sns.set_theme()
np.random.seed(0)

class Environment:
    def __init__(self):
        self.action_size = 10
        self.true_expected_rewards = np.random.uniform(-2, 2, self.action_size) # q*(a)
        print(self.true_expected_rewards)
        self.action_values = [0 for _ in range(self.action_size)]
        self.num_action_selections = [0 for _ in range(self.action_size)]
        self.max_steps = 1000
        # self.data_collector = DataCollector()
        self.reward_per_step = []
        self.epsilon = 0.01

    def evaluate(self, reward, action):
        self.num_action_selections[action] += 1
        self.action_values[action] += (reward - self.action_values[action]) / self.num_action_selections[action]

    def step(self, action):
        prob = np.random.uniform()
        if prob > self.epsilon:
            next_action = np.argmax(self.action_values[action])
        else:
            next_action = np.random.choice(self.action_size)
        reward = np.random.normal(self.true_expected_rewards[next_action], 1)
        # self.data_collector.collect_reward(reward)
        self.reward_per_step.append(reward)
        print(reward)
        return next_action, reward
    
    def run(self):
        action = np.random.choice(self.action_size)
        for step in range(self.max_steps):
            action, reward = self.step(action)
            self.evaluate(reward, action)

    def reward_plot(self):
        average_reward, i = [], 0
        for step in range(self.max_steps):
            i += self.reward_per_step[step]
            average_reward.append(i / (step + 1))
        ax = sns.scatterplot(average_reward)
        plt.show()

# class Agent:
#     def __init__(self):
#         self.trajectory = [[0] for _ in range(10)]
#         self.action_values = [[] for _ in range(10)]
#         # self.action_value = 0
#         self.environment = Environment()

if __name__ == '__main__':
    env = Environment()
    env.run()
    # env.data_collector.reward_plot()
    env.reward_plot()
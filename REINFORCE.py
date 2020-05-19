import argparse
import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="REINFORCE")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--max_steps", type=int, default=500, help="maximum steps")
parser.add_argument("--max_eps", type=int, default=10000, help="maximum episodes")
parser.add_argument("--lr_actor", type=float, default=1e-4, help="actor net lr")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="critic net lr")
parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment")
args = parser.parse_args()

summary = SummaryWriter()

class Net(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        prob = self.fc3(h2)
        prob = F.softmax(prob, dim=1)

        return prob

class Agent:
    def __init__(self, state_size, action_size):
        n_in, n_mid, n_out = state_size, 50, action_size
        self.net = Net(n_in, n_mid, n_out)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.01)

    def get_action_prob(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) # state : [1, 4]
        prob = self.net(state)
        action = prob.multinomial(num_samples=1)
        action = action.item()
        log_prob = torch.log(prob.squeeze(0)[action])

        return action, log_prob

    def update(self, rewards, log_probs):
        self.net.train()
        discounted_rewards = []

        for t in range(len(rewards)):
            discounted_reward = 0
            pw = 0
            for r in rewards[t:]:
                discounted_reward += (args.gamma ** pw) * r
                pw += 1
            discounted_rewards.append(discounted_reward)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for G_t, log_prob in zip(discounted_rewards, log_probs):
            loss.append(-log_prob * G_t)
        self.optimizer.zero_grad()
        loss = torch.stack(loss).mean()

        loss.backward()
        self.optimizer.step()

        return loss.item()

class Environment:
    def __init__(self):
        self.env = gym.make(args.env)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(state_size, action_size)

    def run(self):
        for episode in range(args.max_eps):
            log_probs, rewards = [], []
            total_reward = 0
            state = self.env.reset()

            for step in range(args.max_steps):
                self.env.render()
                action, log_prob = self.agent.get_action_prob(state)
                next_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                total_reward += reward
                state = next_state

                if done:
                    loss = self.agent.update(rewards, log_probs)
                    break

            print("Episode", episode, "||", "Cost", round(loss, 2), "||", "Reward", round(total_reward, 2))
            if episode % 10 == 0:
                summary.add_scalar("loss", loss, episode)
                summary.add_scalar("reward", total_reward, episode)

env = Environment()
env.run()
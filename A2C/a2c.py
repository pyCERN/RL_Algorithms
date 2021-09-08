import argparse
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="a2c")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--max_steps", type=int, default=500, help="maximum steps")
parser.add_argument("--max_eps", type=int, default=10000, help="maximum episodes")
parser.add_argument("--lr_actor", type=float, default=1e-4, help="actor net lr")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="critic net lr")
parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment")
args = parser.parse_args()

summary = SummaryWriter()

class Actor(nn.Module):
    def __init__(self, n_in, n_mid, n_out):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        prob = self.fc3(h2)
        prob = F.softmax(prob, dim=1)

        return prob

class Critic(nn.Module):
    def __init__(self, n_in, n_mid):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        v_value = self.fc3(h2)

        return v_value

class Agent:
    def __init__(self, state_size, action_size):
        n_in, n_mid, n_out = state_size, 50, action_size
        self.actor = Actor(n_in, n_mid, n_out)
        self.critic = Critic(n_in, n_mid)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

    def get_action_prob(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0) # state : [1, 4]
        policy = self.actor(state)
        action = policy.multinomial(num_samples=1)
        action = action.item()
        log_prob = torch.log(policy.squeeze(0)[action])

        return action, log_prob

    def get_v_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)  # state : [1, 4]
        v_value = self.critic(state)

        return v_value

    def update_critic(self, v_value, reward, next_v_value):
        with torch.no_grad():
            td_target = reward + args.gamma * next_v_value
            advantage = td_target - v_value

        loss_critic = 0.5 * (td_target - v_value) ** 2

        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()

        return advantage, loss_critic.item()

    def update_actor(self, log_prob, advantage):
        loss_actor = -log_prob * advantage

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        return loss_actor.item()

class Environment:
    def __init__(self):
        self.env = gym.make(args.env)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(state_size, action_size)

    def run(self):
        for episode in range(args.max_eps):
            state = self.env.reset()
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward = 0

            for step in range(args.max_steps):
                #self.env.render()
                action, log_prob = self.agent.get_action_prob(state)
                next_state, reward, done, _ = self.env.step(action)
                v_value = self.agent.get_v_value(state)
                next_v_value = self.agent.get_v_value(next_state)

                if done:
                    if step < args.max_steps:
                        reward = -100.0

                    next_v_value = 0.0

                advantage, loss_critic = self.agent.update_critic(v_value, reward, next_v_value)
                loss_actor = self.agent.update_actor(log_prob, advantage)
                state = next_state
                total_actor_loss += loss_actor
                total_critic_loss += loss_critic
                total_reward += reward

                if done:
                    break

            print("Episode", episode, "||", "Actor loss", round(total_actor_loss, 2), "||", \
                  "Critic loss", round(total_critic_loss, 2), "||", "Reward", round(total_reward, 2), "||","Steps", step+1)
            if episode % 10 == 0:
                summary.add_scalar("loss_actor", total_actor_loss, episode)
                summary.add_scalar("loss_critic", total_critic_loss, episode)
                summary.add_scalar("reward", total_reward, episode)

env = Environment()
env.run()
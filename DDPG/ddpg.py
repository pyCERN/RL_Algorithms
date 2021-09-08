import argparse
import gym
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from torch.distributions import Normal

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description="ddpg")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--max_eps", type=int, default=10000, help="maximum episodes")
parser.add_argument("--lr_actor", type=float, default=1e-4, help="actor net lr")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="critic net lr")
parser.add_argument("--tau", type=float, default=1e-3, help="soft target update")
parser.add_argument("--mu", type=float, default=0.0, help="ou noise paramater")
parser.add_argument("--rho", type=float, default=0.15, help="ou noise paramater")
parser.add_argument("--sigma", type=float, default=0.2, help="ou noise parameter")
parser.add_argument("--dtime", type=float, default=0.1)
parser.add_argument("--buffer_size", type=int, default=20000)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--env", type=str, default="Pendulum-v0", help="gym environment")
args = parser.parse_args()

summary = SummaryWriter()

class Actor(nn.Module):
    def __init__(self, n_in):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, state):
        h1 = F.relu(self.fc1(state))
        h2 = F.relu(self.fc2(h1))
        action = torch.tanh(self.fc3(h2))

        return action

class Critic(nn.Module):
    def __init__(self, n_in, n_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2_1 = nn.Linear(64, 32)
        self.fc2_2 = nn.Linear(n_action, 32) # action input
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 1)

    def forward(self, state, action):
        h1 = F.relu(self.fc1(state))
        h2_1 = F.relu(self.fc2_1(h1))
        h2_2 = F.relu(self.fc2_2(action))
        h2 = torch.cat((h2_1, h2_2), 1)
        h3 = F.relu(self.fc3(h2))
        q_value = self.fc4(h3)

        return q_value

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add_buffer(self, transition):
        if len(self.memory) >= self.capacity:
            del self.memory[0]
        self.memory.append(transition)

    def sample(self, BATCH_SIZE):
        samples = random.sample(self.memory, BATCH_SIZE)

        states = torch.tensor([samples[i].state.numpy() for i in range(BATCH_SIZE)])
        actions = torch.tensor([[samples[i].action] for i in range(BATCH_SIZE)])
        rewards = torch.tensor([[samples[i].reward] for i in range(BATCH_SIZE)])
        next_states = torch.tensor([samples[i].next_state.numpy() for i in range(BATCH_SIZE)])
        dones = torch.tensor([samples[i].done for i in range(BATCH_SIZE)])

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size):
        self.actor = Actor(state_size)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=args.lr_actor)

        self.target_actor = Actor(state_size)
        self.target_actor_optim = optim.Adam(self.target_actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(state_size, action_size)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.target_critic = Critic(state_size, action_size)
        self.target_critic_optim = optim.Adam(self.target_critic.parameters(), lr=args.lr_critic)

    def get_action(self, state):
        action = self.actor(state)

        return action

    def get_q_value(self, state, action):
        q_value = self.critic(state, action)

        return q_value

    def get_td_target(self, rewards, next_states, dones):
        target_q_values = self.target_critic(next_states, self.target_actor(next_states))
        td_targets = torch.zeros_like(rewards)

        for i in range(rewards.shape[0]):
            if dones[i]:
                td_targets[i] = rewards[i]
            else:
                td_targets[i] = rewards[i] + args.gamma * target_q_values[i]

        return td_targets

    def ou_noise(self, pre_noise, size):
        dist = Normal(0, 1)

        return pre_noise + args.rho * (args.mu - pre_noise) * args.dtime + args.dtime ** 0.5 * args.sigma * dist.sample((size,))

    def update_critic(self, states, actions, rewards, next_states, dones):
        td_targets = self.get_td_target(rewards, next_states, dones)
        q_values = self.get_q_value(states, actions)
        loss = F.smooth_l1_loss(q_values, td_targets)

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.item()

    def update_actor(self, states):
        loss = -self.get_q_value(states, self.get_action(states)).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.item()

    def update_target_net(self):
        actor_param = self.actor.state_dict()
        target_actor_param = self.target_actor.state_dict()
        critic_param = self.critic.state_dict()
        target_critic_param = self.target_critic.state_dict()
        for key, value in actor_param.items():
            target_actor_param[key] = args.tau * value + (1 - args.tau) * target_actor_param[key]
        for key, value in critic_param.items():
            target_critic_param[key] = args.tau * value + (1 - args.tau) * target_critic_param[key]

        self.target_actor.load_state_dict(target_actor_param)
        self.target_critic.load_state_dict(target_critic_param)

class Environment:
    def __init__(self):
        self.env = gym.make(args.env)
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.shape[0]
        self.agent = Agent(self.state_size, self.action_size)
        self.replay = ReplayBuffer(args.buffer_size)

    def run(self):
        Transition = namedtuple("Transition", "state action reward next_state done")

        for episode in range(args.max_eps):
            done = False
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward = 0
            pre_noise = torch.zeros(self.action_size)

            state = self.env.reset()
            state = torch.tensor(state).float()

            while not done:
                self.env.render()
                with torch.no_grad():
                    action = self.agent.get_action(state)
                    noise = self.agent.ou_noise(pre_noise, self.action_size)
                    action = action + noise

                action = torch.clamp(action, -2.0, 2.0).detach()
                next_state, reward, done, _ = self.env.step(action)
                next_state = torch.tensor(next_state).float()
                reward = reward.item()
                transition = Transition(state, action, (reward+8)/8, next_state, done)
                self.replay.add_buffer(transition)

                # 학습 진행
                if self.replay.__len__() > 10000:
                    states, actions, rewards, next_states, dones = self.replay.sample(args.batch_size)
                    loss_critic = self.agent.update_critic(states, actions, rewards, next_states, dones)
                    loss_actor = self.agent.update_actor(states)
                    self.agent.update_target_net()
                    total_actor_loss += loss_actor
                    total_critic_loss += loss_critic

                pre_noise = noise
                state = next_state
                total_reward += reward

            print("Episode", episode, "||", "Actor loss", round(total_actor_loss, 2), "||", \
                  "Critic loss", round(total_critic_loss, 2), "||", "Reward", round(total_reward, 2))
            if episode % 10 == 0:
                summary.add_scalar("loss_actor", total_actor_loss, episode)
                summary.add_scalar("loss_critic", total_critic_loss, episode)
                summary.add_scalar("reward", total_reward, episode)

env = Environment()
env.run()

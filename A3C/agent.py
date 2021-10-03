import torch
import torch.optim as optim

import threading
import multiprocessing
from actor_critic import Actor, Critic


class Agent:
    '''
    Global network
    '''
    def __init__(self, args, state_size, action_size):
        n_in, n_out = state_size, action_size
        self.args = args
        self.actor = Actor(n_in, n_out)
        self.critic = Critic(n_in)
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
        td_target = reward + self.args.gamma * next_v_value
        advantage = td_target - v_value

        loss_critic = 0.5 * (td_target - v_value) ** 2

        self.critic_optim.zero_grad()
        loss_critic.backward(retain_graph=True)
        self.critic_optim.step()

        return advantage, loss_critic.item()

    def update_actor(self, log_prob, advantage):
        loss_actor = -log_prob * advantage

        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()

        return loss_actor.item()


class Worker(threading.Thread):
    '''
    Worker threads
    '''
    def __init__(self):
        threading.Thread.__init__(self)

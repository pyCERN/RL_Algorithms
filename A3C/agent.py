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

    # def update_critic(self, v_value, reward, next_v_value):
    #     td_target = reward + self.args.gamma * next_v_value
    #     advantage = td_target - v_value

    #     loss_critic = 0.5 * advantage ** 2

    #     self.critic_optim.zero_grad()
    #     loss_critic.backward(retain_graph=True)
    #     self.critic_optim.step()

    # def update_actor(self, log_prob, advantage):
    #     loss_actor = -log_prob * advantage

    #     self.actor_optim.zero_grad()
    #     loss_actor.backward()
    #     self.actor_optim.step()

    def update_actor(self, actor_grad):
        for p in self.actor.parameters():
            new_p = p - self.args.lr_actor * actor_grad
            p.copy_(new_p)

    def update_critic(self, critic_grad):
        for p in self.critic.parameters():
            new_p = p - self.args.lr_critic * critic_grad
            p.copy_(new_p)

    def set_param(self, actor_params, critic_params):
        self.actor.load_state_dict(actor_params)
        self.critic.load_state_dict(critic_params)

    def load_params(self, actor_grad, critic_grad):
        self.update_actor(actor_grad)
        self.update_critic(critic_grad)

        return (self.actor.parameters(), self.critic.parameters())


class Worker(threading.Thread):
    '''
    Worker threads
    '''
    def __init__(self, args, global_net, state_size, action_size):
        threading.Thread.__init__(self)

        n_in, n_out = state_size, action_size
        self.args = args
        self.global_net = global_net
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

    def get_critic_grad(self, v_value, reward, next_v_value):
        td_target = reward + self.args.gamma * next_v_value
        advantage = td_target - v_value

        loss_critic = 0.5 * advantage ** 2

        loss_critic.backward(retain_graph=True)

        return loss_critic.grad

    def get_actor_grad(self, log_prob, advantage):
        loss_actor = -log_prob * advantage

        loss_actor.backward()

        return loss_actor.grad
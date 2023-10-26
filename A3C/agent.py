import torch
import torch.optim as optim

import threading
from actor_critic import ActorCritic


class Agent:
    '''
    Global network
    '''
    def __init__(self, args, env):        
        self.args = args
        self.env = env
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        # self.actor = Actor(state_size, action_size)
        # self.critic = Critic(state_size)
        self.actor_critic = ActorCritic(state_size, action_size)

    # def update_actor(self, grad_actor):
    #     for p in self.actor.parameters():
    #         new_p = p - self.args.lr_actor * grad_actor
    #         p.copy_(new_p)

    # def update_critic(self, grad_critic):
    #     for p in self.critic.parameters():
    #         new_p = p - self.args.lr_critic * grad_critic
    #         p.copy_(new_p)
    def update(self, grad_ac):
        for p in self.actor_critic.parameters():
            new_p = p - self.args.lr_ac * grad_ac
            p.copy_(new_p)

    def set_param(self, params_ac):
        self.actor_critic.load_state_dict(params_ac)
        # self.critic.load_state_dict(params_critic)

    def load_params(self, grad_ac):
        self.update(grad_ac)
        # self.update_critic(grad_critic)

        return self.grad_ac.parameters()


class Worker(threading.Thread):
    '''
    Worker threads
    '''
    def __init__(self, args, env, global_net):
        super().__init__()

        self.args = args
        self.env = env
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n

        self.global_net = global_net
        self.actor_critic = ActorCritic(state_size, action_size)
        # self.critic = Critic(state_size)
        self.ac_optim = optim.Adam(self.actor_critic.parameters(), lr=args.lr_ac)
        # self.critic_optim = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

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

    def run(self):
        for episode in range(self.args.max_eps):
            state = self.env.reset()
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward = 0

            for step in range(self.args.max_steps):
                # self.env.render()
                action, log_prob = self.get_action_prob(state)
                next_state, reward, done, _ = self.env.step(action)
                v_value = self.get_v_value(state)
                next_v_value = self.get_v_value(next_state)

                if done:
                    if step < self.args.max_steps:
                        reward = -100.0

                    next_v_value = 0.0

                grad_actor = self.get_actor_grad(log_prob, advantage)
                grad_critic = self.get_critic_grad(v_value, reward, next_v_value)



                advantage, loss_critic = self.update_critic(v_value, reward, next_v_value)
                loss_actor = self.update_actor(log_prob, advantage)
                state = next_state
                total_actor_loss += loss_actor
                total_critic_loss += loss_critic
                total_reward += reward

                if done:
                    break

            print("Episode", episode, "||", "Actor loss", round(total_actor_loss, 2), "||", \
                  "Critic loss", round(total_critic_loss, 2), "||", "Reward", round(total_reward, 2), "||","Steps", step+1)
            # if episode % 10 == 0:
            #     summary.add_scalar("loss_actor", total_actor_loss, episode)
            #     summary.add_scalar("loss_critic", total_critic_loss, episode)
            #     summary.add_scalar("reward", total_reward, episode)

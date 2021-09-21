import gym
import matplotlib.pyplot as plt
from agent import Agent


class Environment:
    def __init__(self, args):
        self.args = args
        self.env = gym.make(args.env)
        state_size = self.env.observation_space.shape[0]
        action_size = self.env.action_space.n
        self.agent = Agent(args, state_size, action_size)

    def run(self):
        for episode in range(self.args.max_eps):
            state = self.env.reset()
            total_actor_loss = 0
            total_critic_loss = 0
            total_reward = 0

            for step in range(self.args.max_steps):
                #self.env.render()
                action, log_prob = self.agent.get_action_prob(state)
                next_state, reward, done, _ = self.env.step(action)
                v_value = self.agent.get_v_value(state)
                next_v_value = self.agent.get_v_value(next_state)

                if done:
                    if step < self.args.max_steps:
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
            # if episode % 10 == 0:
            #     summary.add_scalar("loss_actor", total_actor_loss, episode)
            #     summary.add_scalar("loss_critic", total_critic_loss, episode)
            #     summary.add_scalar("reward", total_reward, episode)
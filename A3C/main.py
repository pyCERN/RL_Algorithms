import argparse
import gym

from agent import Agent, Worker

'''
Gradient Parallelism
'''

parser = argparse.ArgumentParser(description="a2c")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--max_steps", type=int, default=500, help="maximum steps")
parser.add_argument("--max_eps", type=int, default=10000, help="maximum episodes")
parser.add_argument("--lr_ac", type=float, default=1e-4, help="actor_critic net lr")
# parser.add_argument("--lr_critic", type=float, default=1e-3, help="critic net lr")
parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment")
args = parser.parse_args()

num_workers = 2
env = gym.make(args.env)
global_net = Agent(args, env)
workers = [Worker(args, env, global_net) for _ in range(num_workers)]

for worker in workers:
    worker.start()
    worker.join()
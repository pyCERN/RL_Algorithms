import argparse

import torch
import torch.nn as nn

from agent import Agent
from env import Environment

'''
Gradient Parallelism
'''

parser = argparse.ArgumentParser(description="a2c")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--max_steps", type=int, default=500, help="maximum steps")
parser.add_argument("--max_eps", type=int, default=10000, help="maximum episodes")
parser.add_argument("--lr_actor", type=float, default=1e-4, help="actor net lr")
parser.add_argument("--lr_critic", type=float, default=1e-3, help="critic net lr")
parser.add_argument("--env", type=str, default="CartPole-v1", help="gym environment")
args = parser.parse_args()

env = Environment(args)
env.run()
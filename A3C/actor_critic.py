import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    def __init__(self, n_in, n_out):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3_1 = nn.Linear(64, n_out)
        self.fc3_2 = nn.Linear(64, 1)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        prob = self.fc3_1(h2)
        prob = F.softmax(prob, dim=1)
        v_value = self.fc3_2(h2)

        return prob, v_value

# class Critic(nn.Module):
#     def __init__(self, n_in):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(n_in, 64)
#         self.fc2 = nn.Linear(64, 64)
#         self.fc3 = nn.Linear(64, 1)

#     def forward(self, x):
#         h1 = F.relu(self.fc1(x))
#         h2 = F.relu(self.fc2(h1))
#         v_value = self.fc3(h2)

#         return v_value
        
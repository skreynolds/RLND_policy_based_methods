# import requireed libraries
import gym
import math
import numpy as np
from collections import deque
import matplotlib,pyplot as pyplot

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Agent(nn.Module):
	def __init__(self, env, h_size=16):
		
		# inherit the parent class initialisation
		super(Agent, self).__init__()
		
		# spin up an environment as part of the agent initialialisation (why?)
		self.env = env

		# state, hidden layer, actions sizes
		self.s_size = env.observation_space.shape[0]
		self.h_size = h_size
		self.a_size = env.action_space.shape[0]

		# define layers
		self.fc1 = nn.Linear(self.s_size, self.h_size)
		self.fc2 = nn.Linear(self.h_size, self.a_size)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.tanh(self.fc2(x))
		return x.cpu().data

	

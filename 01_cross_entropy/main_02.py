#!/usr/bin/env python 

"""
NOTE: this script should be run in the RoboND environment
"""

# import required libraries
import gym
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# import the agent class
from agent import *

def main():

	#######################################
	# Requred code from main_01.py
	#######################################
	
	# specify the device that will be used to train the agent
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# spin up the environment
	env = gym.make('MountainCarContinuous-v0')
	
	# view environment state and action spaces
	print('observation space:', env.observation_space)
	print('action space:', env.action_space)
	print('   - low:', env.action_space.low)
	print('   - high', env.action_space.high)
	
	# specify seeds to replicability
	env.seed(101)
	np.random.seed(101)

	# spin up the agent
	agent = Agent(env).to(device)

	#######################################
	# View the trained agent performance
	#######################################
	agent.load_state_dict(torch.load('checkpoint.pth'))

	state = env.reset()
	img = plt.imshow(env.render(mode='rgb_array'))
	while True:
		state = torch.from_numpy(state).float().to(device)
		with torch.no_grad():
			action = agent(state)
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		next_state, reward, done, _ = env.step(action)
		state = next_state
		if done:
			break

	env.close()


if __name__ == '__main__':
	main()
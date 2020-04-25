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

# import the cross entropy method monitoring functino
from monitoring import *

def main():

	#######################################
	# Initialise the agent
	#######################################

	# specify the device that will be used to train the agent
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# spin up the environment
	env = gym.make('MountainCarContinuous-v0')
	
	# specify seeds to replicability
	env.seed(101)
	np.random.seed(101)

	# view environment state and action spaces
	print('observation space:', env.observation_space)
	print('action space:', env.action_space)
	print('   - low:', env.action_space.low)
	print('   - high', env.action_space.high)

	# spin up the agent
	agent = Agent(env).to(device)

	#######################################
	# Train the agent
	#######################################

	# train the agent
	scores = cem(agent)

	# plot the scores
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(1, len(scores)+1), scores)
	plt.xlabel('Score')
	plt.ylabel('Episode #')
	plt.show()


if __name__ == '__main__':
	main()
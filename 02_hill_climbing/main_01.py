#!/usr/bin/env python

"""
NOTE: execute this script from the terminal using the RoboND environment
"""

# import required libraries
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# import the policy class
from policy import *

# import the hill cimibing algorithm
from monitoring import *


def main():
	
	############################################
	# Initialisation of environment
	############################################

	# initialise the environment
	env = gym.make('CartPole-v0')
	env.seed(0)
	np.random.seed(0)

	# get a look at the state and aciton spaces
	print('observation space:', env.observation_space)
	print('action space:', env.action_space)

	# initialise the agent policy
	policy = Policy()

	# run the hill climbing algorithm
	scores = hill_climbing(env, policy)

	# plot the scores
	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.plot(np.arange(1, len(scores)+1), scores)
	plt.xlabel('Score')
	plt.ylabel('Episode #')
	plt.show()


if __name__ == '__main__':
	main()
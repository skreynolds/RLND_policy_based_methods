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

	# initialise the environment
	env = gym.make('CartPole-v0')

	# establish empty policy
	policy = Policy()

	# load saved weight to policy
	best_weights = np.load('best_weights.npy')
	policy.w = best_weights

	# capture first state
	state = env.reset()

	# set up redering environment
	img = plt.imshow(env.render(mode='rgb_array'))

	# play episode with smart agent
	for t in range(200):
		action = policy.act(state)
		img.set_data(env.render(mode='rgb_array'))
		plt.axis('off')
		state, reward, done, _ = env.step(action)
		if done:
			break

	env.close()


if __name__ == '__main__':
	main()
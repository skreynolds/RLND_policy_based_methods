# import requred libraries
import numpy as np
from collections import deque
import torch

def cem(agent, n_iterations=500, max_t=1000, gamma=1.0,
		print_every=10, pop_size=50, elite_frac=0.2, sigma=0.5):
		""" PyTorch implementation of cross-entropy method.

		Params
		======
		- n_iterations (int): maximum number of training iterations
		- max_t (int): maximum number of timesteps per episode
		- gamma (float): discount rate
		- print_every (int): how often to print average score (over last 100 episodes)
		- pop__size (int): size of population at each iteration
		- elite_frac (float): percentage of top performers to use in update
		- sigma (float): standard deviation of additive noise
		"""

		n_elite = int(pop_size*elite_frac)

		scores_deque = deque(maxlen=100)
		scores = []

		# initialise best weights with random values
		best_weight = sigma*np.random.randn(agent.get_weights_dim())

		for i_iteration in range(1, n_iterations+1):

			# create a population of weights by adding random noise to existing weights
			weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim())) for i in range(pop_size)]

			# extract the rewards from the perturbed weights population
			rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])

			elite_idxs = rewards.argsort()[-n_elite:] # sclice out the ordered elite fraction (0.2)
			elite_weights = [weights_pop[i] for i in elite_idxs] # capture elite weights
			best_weight =  np.array(elite_weights).mean(axis=0) # take mean of elite weights

			reward = agent.evaluate(best_weight, gamma=1.0)
			scores_deque.append(reward)
			scores.append(reward)

			torch.save(agent.state_dict(), 'checkpoint.pth')

			if i_iteration % print_every == 0:
				print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

			if np.mean(scores_deque)>=90.0:
				print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration-100, np.mean(scores_deque)))
				break

		return scores
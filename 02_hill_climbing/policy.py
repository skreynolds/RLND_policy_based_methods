# import the required libraries
import numpy as np


# define the policy class
class Policy():
	def __init__(self, s_size=4, a_size=2):
		self.w = 1e-4*np.random.rand(s_size, a_size) # weights for a simple linear policy

	def forward(self, state):
		x = np.dot(state, self.w)
		return np.exp(x)/sum(np.exp(x)) # softmax return fuction (for probabilities)

	def act(self, state):
		probs = self.forward(state)
		#action = np.random.choice(2, p=probs)	# option 1: stochastic policy
		action = np.argmax(probs)				# option 2: deterministic policy
		return action
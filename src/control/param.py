from control.rl import *
from control.dqn import *
import numpy as np

class QL1(QL):

	def __init__(self):
		super().__init__()

		self.pvbins	=		11	# Left paddle (me) y position
		self.p2vbins =		11	# Right paddle (opponent) y position
		self.hbins =		11	# Absolute ball x position
		self.vbins =		11	# Absolute ball y position
		self.bsbins =		5	# Ball speed magnitude
		self.pbbins =		5	# Paddle zones
		self.angle_bins =	36	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.3
		self.gamma = 1.0
		self.epsilon = 0.0

class QL2(QL):

	def __init__(self):
		super().__init__()

		self.pvbins	=		5	# Left paddle (me) y position
		self.p2vbins =		5	# Right paddle (opponent) y position
		self.hbins =		5	# Absolute ball x position
		self.vbins =		5	# Absolute ball y position
		self.bsbins =		3	# Ball speed magnitude
		self.pbbins =		3	# Paddle zones
		self.angle_bins =	12	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.4
		self.gamma = 1.0
		self.epsilon = 0.0

class QLd1(QLd):

	def __init__(self):
		super().__init__()

		self.pvbins	=		5	# Left paddle (me) y position
		self.p2vbins =		5	# Right paddle (opponent) y position
		self.hbins =		5	# Absolute ball x position
		self.vbins =		5	# Absolute ball y position
		self.bsbins =		3	# Ball speed magnitude
		self.pbbins =		3	# Paddle zones
		self.angle_bins =	12	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.4
		self.gamma = 1.0
		self.epsilon = 0.0 # Ignored

		# Decay parameters for epsilon
		self.c0 = 0.3
		self.c1 = 0.3
		self.c2 = 3

class QLd2(QLd):

	def __init__(self):
		super().__init__()

		self.pvbins  =		5	 # Left paddle (me) y position
		self.p2vbins =		5    # Right paddle (opponent) y position
		self.hbins =		5	 # Absolute ball x position
		self.vbins =		5	 # Absolute ball y position
		self.bsbins =		3	  # Ball speed magnitude
		self.pbbins =		3	  # Paddle zones
		self.angle_bins =	12    # Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.4
		self.gamma = 1.0
		self.epsilon = 0.0 # Ignored

		# Decay parameters for epsilon
		self.c0 = 0.3
		self.c1 = 0.1
		self.c2 = 3

class QLe1(QLe):

	def __init__(self):
		super().__init__()

		self.pvbins	=		11	# Left paddle (me) y position
		self.p2vbins =		11	# Right paddle (opponent) y position
		self.hbins =		11	# Absolute ball x position
		self.vbins =		11	# Absolute ball y position
		self.bsbins =		5	# Ball speed magnitude
		self.pbbins =		5	# Paddle zones
		self.angle_bins =	36	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.3
		self.gamma = 1.0

		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.1
		self.epsilon_iterations = 2e6

class QLe2(QLe):

	def __init__(self):
		super().__init__()

		self.pvbins	=		5	# Left paddle (me) y position
		self.p2vbins =		5	# Right paddle (opponent) y position
		self.hbins =		5	# Absolute ball x position
		self.vbins =		5	# Absolute ball y position
		self.bsbins =		3	# Ball speed magnitude
		self.pbbins =		3	# Paddle zones
		self.angle_bins =	12	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.4
		self.gamma = 1.0

		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.1
		self.epsilon_iterations = 2e6

class QLe3(QLe2):

	def __init__(self):
		super().__init__()

		self.alpha = 0.5
		self.gamma = 1.0

		self.epsilon = 1.0
		self.epsilon_max = 1.0
		self.epsilon_min = 0.1
		self.epsilon_iterations = 2e6

class SARSA1(SARSA):

	def __init__(self):
		super().__init__()

		self.pvbins	=		11	# Left paddle (me) y position
		self.p2vbins =		11	# Right paddle (opponent) y position
		self.hbins =		11	# Absolute ball x position
		self.vbins =		11	# Absolute ball y position
		self.bsbins =		5	# Ball speed magnitude
		self.pbbins =		5	# Paddle zones
		self.angle_bins =	36	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.3
		self.gamma = 1.0
		self.epsilon = 0.0

class SARSA2(SARSA):

	def __init__(self):
		super().__init__()

		self.pvbins	=		5	# Left paddle (me) y position
		self.p2vbins =		5	# Right paddle (opponent) y position
		self.hbins =		5	# Absolute ball x position
		self.vbins =		5	# Absolute ball y position
		self.bsbins =		3	# Ball speed magnitude
		self.pbbins =		3	# Paddle zones
		self.angle_bins =	12	# Ball direction angle
		#self.angle_bins = int(360 / 60)

		self.num_states = self.pvbins * self.p2vbins * self.vbins * self.hbins * \
			self.bsbins * self.pbbins * self.angle_bins
		self.num_actions = 2
		self.q = np.zeros((self.num_states, self.num_actions))

		self.alpha = 0.4
		self.gamma = 1.0
		self.epsilon = 0.0


class DQN1(DQN):

	def __init__(self):
		super().__init__()

		self.alpha = 0.3
		self.gamma = 0.99
		self.epsilon = 1.0

		self.hidden1 = keras.layers.Dense(256, activation='relu')(self.input)
		self.hidden2 = keras.layers.Dense(256, activation='relu')(self.hidden1)
		#self.hidden3 = keras.layers.Dense(256, activation='relu')(self.hidden2)
		#self.hidden4 = keras.layers.Dense(256, activation='relu')(self.hidden3)

		# Estimated reward for each action
		self.output = keras.layers.Dense(1)(self.hidden2)

		self.model = keras.Model(inputs=self.input, outputs=self.output)

		#self.optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
		#self.optimizer = keras.optimizers.RMSprop(lr=0.5e-4, rho=0.95, epsilon=0.01)
		self.optimizer = keras.optimizers.SGD(lr=1e-4)
		self.model.compile(self.optimizer, loss='mse')
		#self.model.compile(loss='mse', optimizer='sgd')


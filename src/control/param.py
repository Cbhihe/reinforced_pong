from control.rl import *
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
		self.c1 = 1.0
		self.c2 = 3

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



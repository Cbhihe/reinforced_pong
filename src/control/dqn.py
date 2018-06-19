from control.base import Controller

import pygame
import numpy as np
import keras

class DQN(Controller):

	def __init__(self):
		super().__init__()

		#raise NotImplementedError("DQN not ready")

		self.gamma = 1

		self.num_inputs = 6
		self.num_actions = 2

		self.input = keras.Input(
				shape=(self.num_inputs + self.num_actions,))

		self.hidden = keras.layers.Dense(1024)(self.input)

		# Estimated reward for each action
		self.output = keras.layers.Dense(1)(self.hidden)

		self.model = keras.Model(inputs=self.input, outputs=self.output)

		self.optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
		self.model.compile(self.optimizer, loss='mse')

		print(self.model.summary())

		self.a = 0
		self.state = [0] * self.num_inputs

	def reward(self):
		board = self.board
		who = board.get_player_scored(self)
		status = board.get_paddle_status(self)

		if who != None:
			if who == 'me':
				return 1
			else:
				return -1

		if status == 'collision':
			return 0.1

		return -0.001
		#return 0

	def do(self, a):
		#print('Doing '+a)
		board = self.board
		self.doing = a

		down = board.get_paddle_top_speed(self)
		up = -down
		speed = 0

		if a == 0:
			speed = up
		elif a == 1:
			speed = down
			
		board.set_player_speed(self, speed)

		#self.paddle.update()

	def get_state(self):
		b = self.board
		model = self.model

		size = np.array(b.size)
		w,h = size
		topspeed = b.get_ball_top_speed(self)

		bpx, bpy = np.array(b.get_ball_position(self)) / size
		bsx, bsy = np.array(b.get_ball_speed(self)) / topspeed

		p0 = b.get_player_position(self, me=True)  / h
		p1 = b.get_player_position(self, me=False) / h

		return [bpx, bpy, bsx, bsy, p0, p1]

	def update(self):

		model = self.model
		new_state = self.get_state()
		r = self.reward()

		in0 = np.matrix(new_state + [1, 0])
		in1 = np.matrix(new_state + [0, 1])

		predQ = np.array([model.predict(in0), model.predict(in1)])

		maxQ = np.max(predQ)

		out = np.matrix([r + self.gamma * maxQ])

		#print(predQ[0,0])

		# Fit the *previous* state not new_state

		if self.a == 0:
			ins = np.matrix(self.state + [1, 0])
		else:
			ins = np.matrix(self.state + [0, 1])

		h = model.fit(ins, out, verbose=0)
		#print(h.history)



		########################### Now move to t+1 ############################



		state = new_state
		a = np.argmax(predQ)
		self.do(a)
		self.a = a
		self.state = state

		# Wait update from other objects




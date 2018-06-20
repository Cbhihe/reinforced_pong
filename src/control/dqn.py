from control.base import *

import pygame
import numpy as np
import sys,time


class DQN(ControllerLog):

	def __init__(self):
		super().__init__()

		import keras

		#raise NotImplementedError("DQN not ready")

		self.alpha = 0.3
		self.gamma = 0.99
		self.epsilon = 1.0

		self.num_inputs = 6
		self.num_actions = 2

		self.input = keras.Input(
				shape=(self.num_inputs + self.num_actions,))

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
		self.iterations = 0

		#print(self.model.summary())

		self.a = 0
		self.prevQ = 0
		self.state = [0] * self.num_inputs

	def log_header(self):
		print("cputime iteration reward accum_reward mean_reward q0 alpha "+\
				"gamma epsilon points_me points_opp",
			file=self.log_file)

	def log(self):
		# Log all the interesting values

		t = time.clock() - self.start_time # Running time CPU seconds
		iteration = self.iteration

		reward = 0
		mean_reward = 0
		accum_reward = 0

		q0 = 0

		alpha = self.alpha
		gamma = self.gamma
		epsilon = self.epsilon

		points_me = self.board.get_accum_points(self, me=True)
		points_op = self.board.get_accum_points(self, me=False)

		print("{:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {} {}".format(
			t, iteration, reward, accum_reward, mean_reward, q0, alpha, gamma,
			epsilon, points_me, points_op),
			file=self.log_file)

	def action_epsilon_greedy(self, a):
		epsilon = self.epsilon

		if np.random.random() < epsilon:
			# Take random action
			a = np.random.choice(self.num_actions)
			

		return a
	
	def update_epsilon(self):
		ep = self.epsilon
		if ep <= 0.1:
			self.epsilon = 0.1
			return

		maxit = 20000
		ep = (maxit - self.iterations) / maxit
		self.epsilon = ep

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

		#return -0.001
		return 0

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
		self.iterations += 1

		model = self.model
		new_state = self.get_state()
		r = self.reward()
		gamma = self.gamma
		alpha = self.alpha

		in0 = np.matrix(new_state + [1, 0])
		in1 = np.matrix(new_state + [0, 1])

		predQ = np.array([model.predict(in0), model.predict(in1)])

		maxQ = np.max(predQ)
		prevQ = self.prevQ

		out = np.matrix([r + gamma * maxQ])
		#out = np.matrix([prevQ + alpha * (r + gamma * maxQ - prevQ)])

		#print(predQ)

		# Fit the *previous* state not new_state

		if self.a == 0:
			ins = np.matrix(self.state + [1, 0])
		else:
			ins = np.matrix(self.state + [0, 1])

		h = model.fit(ins, out, verbose=0)
		#print(h.history)



		########################### Now move to t+1 ############################



		self.prevQ = maxQ
		state = new_state
		a = np.argmax(predQ)
		a = self.action_epsilon_greedy(a)
		self.do(a)
		self.a = a
		self.state = state

		# Wait update from other objects

		self.update_epsilon()

		super().update()

#	def save(self):
#		raise NotImplementedError()
#		# serialize model to JSON
#		model_json = model.to_json()
#		with open("model.json", "w") as json_file:
#			json_file.write(model_json)
#		# serialize weights to HDF5
#		model.save_weights("model.h5")
#		print("Saved model to disk")




import pygame
import numpy as np

class PaddleController:
	def __init__(self, board, paddle, ball):
		self.board = board
		self.paddle = paddle
		self.ball = ball

	def update(self):
		"Update the state of the paddle by looking at the game state"
		raise NotImplementedError()

	def draw(self):
		"Draw some debug information if neccesary"
		# DO NOT attempt to draw the paddle here!
		pass

class PCFollower(PaddleController):

	def update(self):

		bx,by = self.ball.position
		py = self.paddle.y

		if py < by:
			vy = self.paddle.top_speed
		elif py > by:
			vy = -self.paddle.top_speed
		else:
			vy = 0
	
		self.paddle.set_speed(vy)

class PCKeyboard(PaddleController):

	def update(self):
		vy = self.paddle.vy

		up = -self.paddle.top_speed
		down = self.paddle.top_speed

		events = pygame.event.get()
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP: vy = up
				elif event.key == pygame.K_DOWN: vy = down
				elif event.key == pygame.K_q: exit()

			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_UP and vy == up: vy = 0
				elif event.key == pygame.K_DOWN and vy == down: vy = 0
	
		self.paddle.set_speed(vy)

class PCSimpleLearning(PaddleController):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)

		q = np.random.uniform(size=(3, 3))

		# Normalize action probability so that it sums one
		q = q / q.sum(axis=0)

		self.q = q.T

		#print(self.q)

		self.actions = ['up', 'stop', 'down']
		self.states = ['above', 'middle', 'below']

		self.alpha = 0.01
		self.gamma = 0.8

	def reward(self, s, a):
		if s == 'above':
			if a == 'up': return 1
			else:
				return -1
		elif s == 'below':
			if a == 'down': return 1
			else:
				return -1
		elif s == 'middle':
			if a == 'stop': return 1
			else:
				return -1
		return 0

	def do(self, a):
		#print('Doing '+a)
		down = -self.paddle.top_speed
		up = self.paddle.top_speed

		if a == 'up':
			self.paddle.set_speed(up)
		elif a == 'stop':
			self.paddle.set_speed(0)
		elif a == 'down':
			self.paddle.set_speed(down)

	def update(self):

		#print(self.q)

		state = 0

		bx,by = self.ball.position
		py = self.paddle.y

		if py < by: state = 'above'
		elif py > by: state = 'below'
		else: state = 'middle'

		q = self.q
		s = self.states.index(state)
		#print(np.sum(q[s,:]))
		action = np.random.choice(self.actions, 1, p=q[s,:])
		a = self.actions.index(action)
		#a = np.argmax(q, axis=1)[s]
		action = self.actions[a]

		r = self.reward(state, action)


		# Take the action
		self.do(action)



		alpha = self.alpha
		gamma = self.gamma



		best_estimate = np.argmax(q, axis=1)[s]

		q[s, a] = (1-alpha) * q[s, a] + alpha * (r + gamma * best_estimate)
		q[s, a] = max(q[s, a], 0.0)

		q = (q.T / q.T.sum(axis=0)).T

		self.q = q


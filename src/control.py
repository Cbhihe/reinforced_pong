import pygame
from pygame import Rect
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
		self.paddle.update()

class PCKeyboard(PaddleController):

	def update(self):
		vy = self.paddle.vy

		up = -self.paddle.top_speed
		down = self.paddle.top_speed

		#events = pygame.event.get()
		events = self.board.events
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP: vy = up
				elif event.key == pygame.K_DOWN: vy = down
				elif event.key == pygame.K_q: exit()

			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_UP and vy == up: vy = 0
				elif event.key == pygame.K_DOWN and vy == down: vy = 0
	
		self.paddle.set_speed(vy)
		self.paddle.update()

class PCSimpleLearning(PaddleController):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)

		#q = np.random.uniform(size=(2, 2))
		q = np.zeros((2, 2)) + 1

		# Normalize action probability so that it sums one
		q = q / q.sum(axis=0)

		self.q = q.T

		#print(self.q)

		self.actions = ['up', 'down']
		self.states = ['above', 'below']

		self.alpha = 0.1
		self.gamma = 0.01
		self.doing = 'up'

	def reward(self, s, a):
		who = self.board.player_scored
		me = self.paddle.side
		if who != None:
			if who != me:
				return -1
			#else:
			#	return -1

		if self.paddle.status == 'collision':
			return 1

		

		#if s == 'above':
		#	if a == 'up': return 1
		#	else:
		#		return -1
		#elif s == 'below':
		#	if a == 'down': return 1
		#	else:
		#		return -1
		#elif s == 'middle':
		#	if a == 'stop': return 1
		#	else:
		#		return -1
		#return 0
		return 0

	def do(self, a):
		#print('Doing '+a)
		action = self.actions[a]
		self.doing = action
		down = self.paddle.top_speed
		up = -self.paddle.top_speed

		if action == 'up':
			self.paddle.set_speed(up)
		elif action == 'down':
			self.paddle.set_speed(down)

		self.paddle.update()

	def get_state(self):
		bx, by = self.ball.position
		py = self.paddle.y

		if by < py: return 0
		elif by >= py: return 1

	def update(self):

		#print(self.q)
		s0 = self.get_state()
		me = self.paddle.side

		q = self.q
		#action = np.random.choice(self.actions, 1, p=q[s0,:])
		#a = self.actions.index(action)

		# Select the action with the maximum Q
		a = np.argmax(q, axis=1)[s0]
		r = self.reward(s0, a)


		# Take the action (we move the paddle here!)
		self.do(a)

		s1 = self.get_state()



		alpha = self.alpha
		gamma = self.gamma


		# The best estimate should be the NEXT state after the update
		best_estimate = np.argmax(q, axis=1)[s1]

		prev_qsa = q[s0, a]

		#q[s, a] = (1-alpha) * q[s, a] + alpha * (r + q[s, a])
		if r != 0:
			q[s0, a] = (1-alpha) * q[s0, a] + alpha * (r + gamma * best_estimate)
			#q[s, a] = max(q[s, a], 0.01)

			q = (q.T / q.T.sum(axis=0)).T
		new_qsa = q[s0, a]

		if r!= 0:
			np.set_printoptions(precision=3)
			print("Player {}".format(me))
			#print("At q[{}, {}] from {:.2f} to {:.2f}".format(
			#	s, a, prev_qsa, new_qsa))
			print(self.q)
			print("-------------------------")

		self.q = q

	def draw(self):
		# Draw the action being taken

		a = self.doing
		p = self.paddle.rect
		b = self.board.rect
		w,h = p.size
		side = self.paddle.side

		w_up = h/2 * self.q[0,0]
		h_up = h/2 * self.q[0,1]
		up_rect = Rect((0,0), (w_up, h_up))

		w_down = h/2 * self.q[1,1]
		h_down = h/2 * self.q[1,0]
		down_rect = Rect((0,0), (w_down, h_down))

		if side == 'left':
			up_rect.topleft = b.topleft
			down_rect.bottomleft = b.bottomleft
		else:
			up_rect.topright = b.topright
			down_rect.bottomright = b.bottomright

		col = (255,0,0)
		col_now = (0,255,0)
		col_up = col
		col_down = col

		if a == 'up': col_up = col_now
		elif a == 'down': col_down = col_now

		pygame.draw.rect(self.board.surf, col_up, up_rect)
		pygame.draw.rect(self.board.surf, col_down, down_rect)

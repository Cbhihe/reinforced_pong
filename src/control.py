import pygame
from pygame import Rect
import numpy as np

DEBUG = False

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

class PC1(PaddleController):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)

		q = np.random.uniform(size=(2, 2))
		#q = np.zeros((2, 2)) + 1

		# Normalize action probability so that it sums one
		q = q / q.sum(axis=0)

		self.q = q.T

		#print(self.q)

		self.actions = ['up', 'down']

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

		#if self.paddle.status == 'collision':
		#	return 1

		

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

		if r!= 0 and DEBUG:
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







class PC2(PaddleController):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)

		self.pvbins = 1
		self.hbins = 1
		self.vbins = 1
		self.pbbins = 3 + 2

		self.num_states = self.pvbins * self.vbins * self.hbins * self.pbbins

		q = np.random.uniform(size=(self.num_states, 2))
		#q = np.zeros((self.num_states, 2)) + 1

		# Normalize action probability so that it sums one
		q = (q.T / q.T.sum(axis=0)).T

		self.q = q

		#print(self.q)

		self.alpha = 0.2
		self.gamma = 0.2
		self.doing = 0
		self.state = 0

	def reward(self, s, a):
		who = self.board.player_scored
		me = self.paddle.side
		if who != None:
			if who != me:
				return -1
			#else:
			#	return 1

		#if self.paddle.status == 'collision':
		#	return 1

		

		return 0

	def do(self, a):
		#print('Doing '+a)
		self.doing = a
		down = self.paddle.top_speed
		up = -self.paddle.top_speed

		if a == 0:
			self.paddle.set_speed(up)
		elif a == 1:
			self.paddle.set_speed(down)

		self.paddle.update()

	def get_state(self):
		bx, by = self.ball.position
		pcentery = self.paddle.rect.centery
		pcenterx = self.paddle.rect.centerx
		ptop = self.paddle.rect.top
		pbottom = self.paddle.rect.bottom

		# In which bin the paddle is located?
		w,h = self.board.size

		pvbin_h = h/self.pvbins
		pvbin = int(pcentery / pvbin_h)
		pvbin = max(0, min(self.pvbins-1,pvbin)) 

		hbin_w = w/self.hbins
		hbin = int(bx / hbin_w)
		hbin = max(0, min(self.hbins-1,hbin)) 

		vbin_h = h/self.vbins
		vbin = int(by / vbin_h)
		vbin = max(0, min(self.vbins-1,vbin)) 

		if by < ptop:
			pbbin = 0
		elif by > pbottom:
			pbbin = self.pbbins - 1
		else:
			pw,ph = self.paddle.size

			pbbin_h = ph / (self.pbbins - 2)
			pbbin = 1 + int((by - ptop) / pbbin_h)
			#print("ball at {}, ptop = {} pbin_h = {} pbin = {}".format(
			#	(bx, by), ptop, pbin_h, pbin))

		#print(vbin, hbin, pbin)

		return (pvbin * self.hbins * self.vbins * self.pbbins) + \
			(hbin * self.vbins * self.pbbins) + \
			(vbin * self.pbbins) + \
			pbbin
			

	def update(self):

		#print(self.q)
		s0 = self.get_state()
		self.state = s0
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
		if r != 0:# or s0 != s1:
		#if True:
			q[s0, a] = (1-alpha) * q[s0, a] + \
					alpha * (r + gamma * best_estimate - q[s0,a])
			#q[s, a] = max(q[s, a], 0.01)

			q = (q.T / q.T.sum(axis=0)).T
		new_qsa = q[s0, a]

		if r!= 0 and DEBUG:
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
		bw,bh = b.size
		pw,ph = p.size
		col = (100,0,0)

		n = self.num_states
		hbar = max(1, bh/n)
		wbar = 40
		for st in range(self.num_states):
			if st == self.state:
				color = (255,0,0)
			else:
				color = col
			x = p.x + pw + 5
			y = st * hbar
			max_a = np.argmax(self.q[st,:])

			ws = max(1, int(wbar * self.q[st, 1]))
			wm = int(wbar * max_a)
			rect = Rect((x, y), (wm, hbar))
			pygame.draw.rect(self.board.surf, (50,0,0), rect)
			rect = Rect((x, y), (ws, hbar))
			pygame.draw.rect(self.board.surf, color, rect)


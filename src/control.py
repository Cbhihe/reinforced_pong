import pygame
from pygame import Rect
import numpy as np
import collections, time

DEBUG = False

class PaddleController:
	def __init__(self):
		self.board = None

	def set_board(self, board):
		self.board = board

	def save(self):
		"Save the status of the controller"
		return None

	def restore(self, data):
		"Restore the status of the controller"
		pass

	def update(self):
		"Update the state of the paddle by looking at the game state"
		raise NotImplementedError()

	def draw(self):
		"Draw some debug information if neccesary"
		# DO NOT attempt to draw the paddle here!
		pass

class PCFollower(PaddleController):

	def update(self):

		board = self.board
		bx, by = board.get_ball_position(self)
		py = board.get_player_position(self)
		top_speed = board.get_paddle_top_speed(self)

		if py < by:
			vy = top_speed
		elif py > by:
			vy = - top_speed
		else:
			vy = 0
	
		board.set_player_speed(self, vy)

class PCKeyboard(PaddleController):

	def update(self):
		board = self.board
		vy = board.get_player_speed(self)

		top = board.get_paddle_top_speed(self)
		up = -top
		down = top

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
	
		board.set_player_speed(self, vy)


class PC2(PaddleController):

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

		print('Number of states is {}'.format(self.num_states))

		#q = np.random.uniform(size=(self.num_states, 2))
		q = np.zeros((self.num_states, self.num_actions))

		# Normalize action probability so that it sums one
		#q = (q.T / q.T.sum(axis=0)).T

		self.q = q

		#print(self.q)

		self.alpha = 0.3
		self.gamma = 1.0
		self.epsilon = 0.0

		self.doing = 0
		self.state = 0
		self.iteration = 0

		self.accum_r = 0.0
		self.last_rewards = collections.deque(maxlen=500000)
		self.last_states = collections.deque(maxlen=30)
		self.s0 = 0
		self.a = 0
		self.print_info = False
		self.should_draw = False
		self.interval_print = True
		self.print_iteration = 0
		self.print_step = 5000
		self.tic = 0

	def save(self):
		t = (self.pvbins, self.p2vbins, self.hbins, self.vbins, self.bsbins,
				self.pbbins, self.angle_bins, self.num_states, self.num_actions,
				self.q, self.alpha, self.gamma, self.epsilon)
		return t

	def restore(self, t):
		(self.pvbins, self.p2vbins, self.hbins, self.vbins, self.bsbins,
			self.pbbins, self.angle_bins, self.num_states, self.num_actions,
			self.q, self.alpha, self.gamma, self.epsilon) = t

	def reward(self, s, a):
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

	def state_vector(self, states, bins):
		s = 0
		last_bins = 1
		for i in range(len(states)-1, -1, -1):
			s += states[i] * last_bins
			last_bins *= bins[i]

		return s

	def get_interval(self, raw_value, steps, min_value=0, max_value=1):
		interval_size = (max_value - min_value)/steps
		interval = int(min_value + (raw_value - min_value) / interval_size)
		interval = max(0, min(steps - 1, interval))

		return interval


	def get_state(self):
		board = self.board
		bx, by = board.get_ball_position(self)
		ball_speed = board.get_ball_speed(self)
		ball_top_speed = board.get_ball_top_speed(self)
		bs = np.linalg.norm(ball_speed)

		pcentery = board.get_player_position(self)

		paddle_rect = board.get_paddle_rect(self)
		ptop = paddle_rect.top
		pbottom = paddle_rect.bottom

		# FIXME choose a better name for these variables
		p2centery = self.board.get_player_position(self, me=False)

		pvbins = self.pvbins
		hbins = self.hbins
		p2vbins = self.p2vbins
		bsbins = self.bsbins
		angle_bins = self.angle_bins
		vbins = self.vbins
		pbbins = self.pbbins

		v = board.get_ball_speed(self)
		vx, vy = v

		# In which bin the paddle is located?
		w,h = self.board.size

		pvbin = self.get_interval(pcentery, pvbins, 0, h)
		p2vbin = self.get_interval(p2centery, p2vbins, 0, h)
		hbin = self.get_interval(bx, hbins, 0, w)
		vbin = self.get_interval(bx, vbins, 0, h)
		bsbin = self.get_interval(bs, bsbins, 0, ball_top_speed)

		angle = np.arctan2(-vy, vx)

		if angle < 0:
			angle += 2*np.pi

		angle_bin = self.get_interval(angle, angle_bins, 0, 2*np.pi)

		if pbbins == 1:
			pbbin = 0
		else:
			if by < ptop:
				pbbin = 0
			elif by > pbottom:
				pbbin = pbbins - 1
			else:
				pw,ph = paddle_rect.size

				pbbin_h = ph / (pbbins - 2)
				pbbin = 1 + int((by - ptop) / pbbin_h)
				#print("ball at {}, ptop = {} pbin_h = {} pbin = {}".format(
				#	(bx, by), ptop, pbin_h, pbin))

		#print(vbin, hbin, pbin)

		s = self.state_vector(
			[pvbin,  hbin,  p2vbin,  bsbin,  angle_bin,  vbin,  pbbin ],
			[pvbins, hbins, p2vbins, bsbins, angle_bins, vbins, pbbins])


		return s
			

	def action_epsilon_greedy(self, state):
		epsilon = self.epsilon

		if np.random.random() < epsilon:
			# Take random action
			a = np.random.choice(self.num_actions)
		else:
			#Take the best action
			a = np.argmax(self.q[state, :])

		return a

	def update(self):

		self.iteration += 1
		
		# Finish last iteration first
		s0 = self.s0
		a = self.a
		alpha = self.alpha
		gamma = self.gamma
		q = self.q

		s1 = self.get_state()
		if not s1 in self.last_states:
			self.last_states.append(s1)
		r = self.reward(s1, a)
		self.accum_r += r
		self.last_rewards.append(r)
		q_max = np.max(q[s1, :])

		# Update q
		should_print = ((r != 0) or (s0 != s1))

		speed = 0
		if self.print_iteration < self.iteration:
			self.print_iteration += self.print_step
			self.interval_print = True
			toc = time.clock()
			elapsed = toc - self.tic
			speed = self.print_step / elapsed
			self.tic = time.clock()


		for event in self.board.events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_i:
					self.print_info = True
				if event.key == pygame.K_d:
					self.should_draw = not self.should_draw


		#prev_q = q.copy()
		np.set_printoptions(precision=3)

		if DEBUG and should_print:
			print("---------- Q is being updated -----------")
			print(q)
			print("Action taken is {}".format(a))
			print("Reward = {}".format(r))
			print("Previous q[s0={}, a={}] = {}".format(s0, a, q[s0, a]))
			print("And max q[s1={}, :] = {}".format(s1, q_max))

		#prev_q_s0_a = q[s0, a]
		q[s0, a] = q[s0, a] + alpha * (r + gamma * q_max - q[s0, a])
		self.q = q

		#diff = q[s0, a] - prev_q_s0_a

		#diff_q = q - prev_q
		#diff = np.linalg.norm(diff_q) / np.linalg.norm(q)
		#diff = np.linalg.norm(diff_q)

		if DEBUG and should_print:
			print("Finally q[s0={}, a={}] = {}".format(s0, a, q[s0, a]))
			print(q)
			print("------------- End of update -------------")
		if should_print and (self.print_info or self.interval_print):
			self.print_info = False
			self.interval_print = False
			mean_r = np.mean(self.last_rewards)
			print("r {:.3e} ({:.3e}), alpha {:.3e} iter {:.2e} ({:.1f} i/s)".format(
				self.accum_r, mean_r, self.alpha, self.iteration, speed))

		# Shift time: t <- t+1

		# Update old state
		self.s0 = s1

		# New action
		self.a = self.action_epsilon_greedy(s1)

		# Take the action (we move the paddle here!)
		# XXX Dont move the paddle
		self.do(self.a)

		self.alpha *= (1 - 1e-8)

		#print("Alpha = {:e}".format(self.alpha))

	def draw(self):
		# Draw the action being taken

		board = self.board
		a = self.doing
		p = board.get_paddle_rect(self)
		b = board.rect
		bw,bh = b.size
		pw,ph = p.size
		col = (200,200,0)

		n = self.num_states
		hbar = max(1, bh/n)

		# Draw the actual state little mark
		dz = board.get_debug_zone(self)
		dw, dh = dz.size
		wbar = dw / 2 - 3
		xc = dz.centerx
		x0 = dz.x
		y0 = int(self.s0 * bh/n)
		rect = Rect((x0, y0), (dw, hbar))
		pygame.draw.rect(self.board.surf, (255,255,0), rect)

		if not self.should_draw:
			return


#		if not self.should_draw:
#			return
#			states = [self.s0]
#			scale = np.max(np.abs(self.q[self.s0,:]))
#		else:
		#states = range(self.num_states)
		states = self.last_states
		scale = np.max(np.abs(self.q[states, :]))

		for st in states:
			#if st != self.s0: continue

			# Skip if the q was not updated (not reached)
			if np.all(self.q[st,:] == 0):
				continue

			x = xc
			y = int(st * bh/n)

			q = self.q
			max_a = np.argmax(self.q[st,:])

			q0 = q[st, 0]
			q1 = q[st, 1]
			qa = q[st, max_a]

			if max_a == 0:
				X = x-3
				w0 = int(np.abs(wbar * q0 / scale))
				W = -w0
			else:
				X = x+3
				w1 = int(np.abs(wbar * q1 / scale))
				W = w1

			# Red if q is negative, green positive
			if qa < 0: color = (255, 0, 0)
			else: color = (0, 255, 0)

			rect_bar = Rect((X, y), (W, hbar))
			pygame.draw.rect(self.board.surf, color, rect_bar)


class PCPredictor(PaddleController):

	def __init__(self):
		super().__init__()
		self.pc = (0,0)

	def predict_collision(self):


		# We try to compute the point in which the ball will collide with the
		# paddle.

		board = self.board

		v = board.get_ball_speed(self)
		b = board.get_ball_position(self)
		p = board.get_paddle_rect(self).topright

		vx, vy = v
		bx, by = b
		px, py = p
		w, h = self.board.size

		# XXX Move this logic to the other side

		#if self.paddle.side != 'right':
		#	print("Only for right players")
		#	raise NotImplementedError()

		# If the ball is going towards the opponent, move the paddle to the
		# center

		if vx >= 0:
			return h/2


		# Otherwise compute the collision point
		while True:

			# Note that vy grows downwards!

			#angle = np.arctan(-vy/vx)
			#print(vx, vy)

			# Ball going upwards, chech top boundary
			if vy < 0:

				dx = by * vx / (-vy)
				#print(dx)

				# If we passed the paddle, abort the bounce
				if bx + dx <= px:
					break

				bx += dx
				by = 0
				vy = -vy

			# Downwards
			elif vy > 0:

				dx = (h - by) * vx / vy

				# If we passed the paddle, abort the bounce
				if bx + dx <= px:
					break

				bx += dx
				by = h
				vy = -vy

		#angle = np.arctan(-vy/vx)

		if bx > px:
			# Now advance the ball to collide with the paddle

			# Horizontal movement (rare)
			if vy == 0:
				bx = px

			# Upwards
			elif vy < 0:
				dy = -vy * (px - bx) / vx
				bx = px
				by -= dy

			# Downwards
			else:
				dy = vy * (px - bx) / vx
				bx = px
				by += dy
		bx = px

		self.pc = (int(bx), int(by))
		#print("Predicted collision {}".format(self.pc))
		return by


	def update(self):

		board = self.board
		pred = self.predict_collision()
		py = board.get_player_position(self)

		paddle_top = board.get_paddle_top_speed(self)

		vy = 0
	
		if py < pred:
			vy = paddle_top
		elif py > pred:
			vy = -paddle_top
	
		board.set_player_speed(self, vy)

	def draw(self):

		board = self.board
		dz = board.get_debug_zone(self)
		x = dz.centerx
		cx, cy = self.pc
		rect = Rect((x - 5, cy - 5), (10, 10))
		color = (100,0,0)
		#pygame.draw.rect(self.board.surf, (0,25,0), dz)
		pygame.draw.rect(self.board.surf, color, rect)


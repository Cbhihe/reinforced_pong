import pygame
from pygame import Rect
import numpy as np
import collections, time

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
		#q = q / q.sum(axis=0)

		self.q = q

		#print(self.q)

		self.actions = ['up', 'down']

		self.alpha = 0.1
		self.gamma = 0.5
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

		# Above
		if by < py: return 0

		# Below
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
		best_estimate = np.argmax(q[s1, :])

		prev_qsa = q[s0, a]

		#q[s, a] = (1-alpha) * q[s, a] + alpha * (r + q[s, a])
		if r != 0:
			q[s0, a] = (1-alpha) * q[s0, a] + alpha * (r + gamma * best_estimate)
			#q[s, a] = max(q[s, a], 0.01)

			#q = (q.T / q.T.sum(axis=0)).T
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

		sum_q0 = np.sum(self.q[0,:])
		sum_q1 = np.sum(self.q[1,:])

		w_up = h/2 * self.q[0,0]/sum_q0
		h_up = h/2 * self.q[0,1]/sum_q0
		up_rect = Rect((0,0), (w_up, h_up))

		w_down = h/2 * self.q[1,1]/sum_q1
		h_down = h/2 * self.q[1,0]/sum_q1
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

		self.pvbins	=		9	# Left paddle (me) y position
		self.p2vbins =		7	# Right paddle (opponent) y position
		self.hbins =		11	# Absolute ball x position
		self.vbins =		11	# Absolute ball y position
		self.bsbins =		5	# Ball speed magnitude
		self.pbbins =		5	# Paddle zones
		self.angle_bins =	18	# Ball direction angle

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
		self.last_states = collections.deque(maxlen=1000)
		self.s0 = 0
		self.a = 0
		self.print_info = False
		self.should_draw = False
		self.interval_print = True
		self.print_iteration = 0
		self.print_step = 5000
		self.tic = 0

	def reward(self, s, a):
		who = self.board.player_scored
		me = self.paddle.side
		if who != None:
			if who != me:
				return -1
			else:
				return 1

		if self.paddle.status == 'collision':
			return 0.1

		return -0.001
		#return 0

	def do(self, a):
		#print('Doing '+a)
		self.doing = a
		down = self.paddle.top_speed
		up = -self.paddle.top_speed

		if a == 0:
			self.paddle.set_speed(up)
		elif a == 1:
			self.paddle.set_speed(down)

		#self.paddle.update()

	def state_vector(self, states, bins):
		s = 0
		last_bins = 1
		for i in range(len(states)-1, -1, -1):
			s += states[i] * last_bins
			last_bins *= bins[i]

		return s

	def get_state(self):
		bx, by = self.ball.position
		bs = np.linalg.norm(self.ball.speed)
		pcentery = self.paddle.rect.centery
		pcenterx = self.paddle.rect.centerx
		ptop = self.paddle.rect.top
		pbottom = self.paddle.rect.bottom

		p2centery = self.board.pr.rect.centery
		p2centerx = self.board.pr.rect.centerx

		pvbins = self.pvbins
		hbins = self.hbins
		p2vbins = self.p2vbins
		bsbins = self.bsbins
		angle_bins = self.angle_bins
		vbins = self.vbins
		pbbins = self.pbbins

		v = self.ball.speed
		vx, vy = v

		# In which bin the paddle is located?
		w,h = self.board.size

		pvbin_h = h/pvbins
		pvbin = int(pcentery / pvbin_h)
		pvbin = max(0, min(pvbins-1,pvbin)) 

		p2vbin_h = h/p2vbins
		p2vbin = int(p2centery / p2vbin_h)
		p2vbin = max(0, min(p2vbins-1,p2vbin)) 

		hbin_w = w/hbins
		hbin = int(bx / hbin_w)
		hbin = max(0, min(hbins-1,hbin)) 

		vbin_h = h/vbins
		vbin = int(by / vbin_h)
		vbin = max(0, min(vbins-1,vbin)) 

		bsbin_h = self.ball.top_speed/bsbins
		bsbin = int(bs / bsbin_h)
		bsbin = max(0, min(bsbins-1,bsbin)) 

		angle = np.arctan2(-vy, vx)
		if angle < 0:
			angle += 2*np.pi
		angle_bin_step = (np.pi*2) / angle_bins

		angle_bin = int(angle / angle_bin_step)
		#angle_bin = max(0, min(self.angle_bins - 1, angle_bin)) 

		#print(angle_bin)

		if pbbins == 1:
			pbbin = 0
		else:
			if by < ptop:
				pbbin = 0
			elif by > pbottom:
				pbbin = pbbins - 1
			else:
				pw,ph = self.paddle.size

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
		me = self.paddle.side

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
			print("---------- Q is being updated -----------".format(me))
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
			print("------------- End of update -------------".format(me))
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

		a = self.doing
		p = self.paddle.rect
		b = self.board.rect
		bw,bh = b.size
		pw,ph = p.size
		col = (200,200,0)

		n = self.num_states
		hbar = max(1, bh/n)
		wbar = 40

		# Draw the actual state little mark
		x0 = p.x
		y0 = int(self.s0 * bh/n)
		rect = Rect((x0 - 7, y0), (5, hbar))
		pygame.draw.rect(self.board.surf, (255,0,0), rect)

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

			x = p.x + pw + 5
			y = int(st * bh/n)

			q = self.q
			max_a = np.argmax(self.q[st,:])

			q0 = q[st, 0]
			q1 = q[st, 1]
			qa = q[st, max_a]

			if max_a == 0:
				X = x+wbar
				w0 = int(np.abs(wbar * q0 / scale))
				W = -w0
			else:
				X = 2+x+wbar
				w1 = int(np.abs(wbar * q1 / scale))
				W = w1

			# Red if q is negative, green positive
			if qa < 0: color = (255, 0, 0)
			else: color = (0, 255, 0)

			rect_bar = Rect((X, y), (W, hbar))
			pygame.draw.rect(self.board.surf, color, rect_bar)


class PCPredictor(PaddleController):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)
		self.pc = (0,0)

	def predict_collision(self):

		# We try to compute the point in which the ball will collide with the
		# paddle.

		v = self.ball.speed
		b = self.ball.position
		p = self.paddle.rect.topleft #XXX Only for right player

		vx, vy = v
		bx, by = b
		px, py = p
		w,h = self.board.size

		if self.paddle.side != 'right':
			print("Only for right players")
			raise NotImplementedError()

		while True:

			# Note that vy grows downwards!

			angle = np.arctan(-vy/vx)

			# Ball going upwards, chech top boundary
			if angle > 0:

				dx = by * vx / (-vy)
				#print(dx)

				# If we passed the paddle, abort the bounce
				if bx + dx >= px:
					break

				bx += dx
				by = 0
				vy = -vy

			# Downwards
			elif angle < 0:

				dx = (h - by) * vx / vy

				# If we passed the paddle, abort the bounce
				if bx + dx >= px:
					break

				bx += dx
				by = h
				vy = -vy

		angle = np.arctan(-vy/vx)

		if bx < px:
			# Now advance the ball to collide with the paddle

			# Easy
			if angle == 0:
				bx = px

			elif angle > 0:
				# If we passed the paddle, abort the bounce
				dy = -vy * (px - bx) / vx
				bx = px
				by -= dy
			else:
				# If we passed the paddle, abort the bounce
				dy = vy * (px - bx) / vx
				bx = px
				by += dy
		bx = px

		self.pc = (int(bx), int(by))
		#print("Predicted collision {}".format(self.pc))


	def update(self):

		self.predict_collision()

		w,h = self.board.size
		bx,by = self.ball.position
		vx,vy = self.ball.speed
		py = self.paddle.y
		_, pred = self.pc
	
		# Ball going left
		if vx < 0:
			pred = h/2

		if py < pred:
			vy = self.paddle.top_speed
		elif py > pred:
			vy = -self.paddle.top_speed
		else:
			vy = 0
	
		self.paddle.set_speed(vy)
		self.paddle.update()

	def draw(self):

		p = self.paddle.rect
		cx, cy = self.pc
		rect = Rect((cx - 10, cy - 5), (10, 10))
		color = (50,0,0)
		pygame.draw.rect(self.board.surf, color, rect)


class PCPredictorLearn(PCPredictor):

	def __init__(self, board, paddle, ball):
		super().__init__(board, paddle, ball)

		# Discretize ball angle (only towards the player)
		self.angle_nbins = 1
		self.pred_nbins = 5
		self.paddle_nbins = 1
		self.num_states = self.angle_nbins * self.pred_nbins * self.paddle_nbins
		self.num_actions = self.pred_nbins

		self.q = np.zeros((self.num_states, self.num_actions)) + 1
		#self.q = np.random.uniform(size=(self.num_states, self.num_actions))
		self.alpha = 0.1
		self.gamma = 0.3
		self.doing = 0

		self.state = 0
		self.target_bin = 0
		self.last_state = -1

	def get_state(self):

		w,h = self.board.size
		pcx, pcy = self.pc
		px, py = self.paddle.rect.topleft
		vx, vy = self.ball.speed

		pred_bin_h = h / self.pred_nbins 
		pred_bin = int(pcy / pred_bin_h)
		pred_bin = max(0, min(self.pred_nbins - 1, pred_bin)) 

		angle = np.arctan(-vy/vx)
		angle_bin_step = np.pi / self.angle_nbins

		# Angle to [0, pi]
		angle += np.pi/2
		angle_bin = int(angle / angle_bin_step)
		angle_bin = max(0, min(self.angle_nbins - 1, angle_bin)) 

		paddle_bin_h = h / self.paddle_nbins
		paddle_bin = int(py / paddle_bin_h)
		paddle_bin = max(0, min(self.paddle_nbins - 1, paddle_bin))

		return pred_bin * self.angle_nbins * self.paddle_nbins + \
				angle_bin * self.paddle_nbins + \
				paddle_bin

	def reward(self, s, a):
		who = self.board.player_scored
		me = self.paddle.side
		if who != None:
			if who != me:
				return -1
			#else:
			#	return 1

		if self.paddle.status == 'collision':
			return 1

		return 0

	def do(self, a):
		#print('Doing '+a)
		#self.doing = a
		down = self.paddle.top_speed
		up = -self.paddle.top_speed

		self.target_bin = a
		update = (self.state != self.last_state)

#		if update:

#			print('Action = {}'.format(a))
#			if a == 0:
#				self.target_bin = max(0, self.target_bin - 1)
#			elif a == 1:
#				self.target_bin = min(self.paddle_nbins-1, self.target_bin + 1)
#			elif a == 2:
#				# target_bin is unchanged
#				pass
#			print(self.target_bin)


		w,h = self.board.size
		px, py = self.paddle.rect.topleft
		#paddle_bin_h = h / self.paddle_nbins
		#paddle_bin = int(py / paddle_bin_h)
		#paddle_bin = max(0, min(self.paddle_nbins - 1, paddle_bin))

		pred_bin_h = h / self.pred_nbins 
		pred_bin = int(py / pred_bin_h)
		pred_bin = max(0, min(self.pred_nbins - 1, pred_bin)) 

		paddle_bin = pred_bin

	#	if update:
	#		print('Paddle is at bin {}, target is {}'.format(paddle_bin, self.target_bin))

		#print('Paddle is at bin {}, target is {}'.format(paddle_bin, self.target_bin))

		if paddle_bin < self.target_bin:
			self.paddle.set_speed(down)
		elif paddle_bin > self.target_bin:
			self.paddle.set_speed(up)
		else:
			self.paddle.set_speed(0)

		#print('Paddle speed = {}'.format(self.paddle.vy))

		self.paddle.update()

	def update(self):

		self.predict_collision()

		q = self.q

		s = self.get_state()
		self.state = s

		a = np.argmax(q, axis=1)[s]
		r = self.reward(s, a)
		self.do(a)

		s1 = self.get_state()
		self.last_state = s1

		# Wait until we reach another state to apply the learning step
		if s == s1 and r == 0: return

		alpha = self.alpha
		gamma = self.gamma

		# The best estimate should be the NEXT state after the update
		qmax = np.argmax(q, axis=1)[s1]

		#q[s, a] = (1-alpha) * q[s, a] + alpha * (r + q[s, a])
		if r != 0:
		#if r != 0 or s0 != s1:
		#if True:
			q[s, a] = (1-alpha) * q[s, a] + alpha * (r + gamma * qmax - q[s,a])
			#q[s, a] = max(q[s, a], 0.01)

			#q = (q.T / q.T.sum(axis=0)).T


			self.q = q

	def draw(self):
		super().draw()
		# Draw the action being taken

		#a = self.doing
		p = self.paddle.rect
		b = self.board.rect
		bw,bh = b.size
		pw,ph = p.size
		col = (70,0,0)

		n = self.num_states
		hbar = max(1, bh/n)
		wbar = 40
		for st in range(self.num_states):
			if st == self.state:
				color = (150,0,0)
			else:
				color = col
			x = p.x + pw + 5
			y = st * hbar
			max_a = np.argmax(self.q[st,:])
			na = self.num_actions
			ws = int(wbar * max_a / na)

			wsum = np.sum(self.q[st, :])
			wm = int(wbar * max_a)
			
			#ws = max(1, int(wbar * self.q[st, 1]/wsum))

			rect = Rect((x, y), (wm, hbar))
			pygame.draw.rect(self.board.surf, (30,0,0), rect)
			rect = Rect((x, y), (ws, hbar))
			pygame.draw.rect(self.board.surf, color, rect)

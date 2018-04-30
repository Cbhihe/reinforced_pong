import pygame
from pygame import Rect
import numpy as np

DEBUG = True

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

		self.pvbins = 1
		self.hbins = 1
		self.vbins = 1
		self.pbbins = 1 + 2

		self.num_states = self.pvbins * self.vbins * self.hbins * self.pbbins

		#q = np.random.uniform(size=(self.num_states, 2))
		q = np.zeros((self.num_states, 2)) + 1

		# Normalize action probability so that it sums one
		#q = (q.T / q.T.sum(axis=0)).T

		self.q = q

		#print(self.q)

		self.alpha = 0.01
		self.gamma = 1.0
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

		if self.paddle.status == 'collision':
			return 1

		

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
		a = np.argmax(q[s0, :])

		r = self.reward(s0, a)

		# Take the action (we move the paddle here!)
		self.do(a)
		

		s1 = self.get_state()



		alpha = self.alpha
		gamma = self.gamma


		# The best estimate should be the NEXT state after the update
		best_estimate = np.max(q[s1, :])

		#prev_qsa = q[s0, a]

		#q[s, a] = (1-alpha) * q[s, a] + alpha * (r + q[s, a])
		#if r != 0:
		#if r != 0 or s0 != s1:
		#if True:
		np.set_printoptions(precision=3)
		if DEBUG:
			print("---------- Q is being updated -----------".format(me))
			print(q)
			print("Action taken is {}".format(a))
			print("Reward = {}".format(r))
			print("Previous q[s0, a] = {}".format(q[s0, a]))
			print("And max q[s1, :] = {}".format(best_estimate))

		q[s0, a] = (1-alpha) * q[s0, a] + \
				alpha * (r + gamma * best_estimate - q[s0, a])

		if DEBUG:
			print("Finally q[s0, a] = {}".format(q[s0, a]))
			print(q)
			print("------------- End of update -------------".format(me))
		#q[s, a] = max(q[s, a], 0.01)

		#q = (q.T / q.T.sum(axis=0)).T
		#new_qsa = q[s0, a]

		#if r!= 0 and DEBUG:
		#	np.set_printoptions(precision=3)
		#	#print("At q[{}, {}] from {:.2f} to {:.2f}".format(
		#	#	s, a, prev_qsa, new_qsa))
		#	print(self.q)
		#	print("-------------------------")

		self.q = q

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

		scale = np.max(np.abs(self.q))

		for st in range(self.num_states):
			if st == self.state:
				color = (255,0,0)
			else:
				color = col
			x = p.x + pw + 5
			y = st * hbar

			q = self.q
			q0 = q[st, 0]
			q1 = q[st, 1]

			w0 = int(np.abs(wbar * q0 / scale))
			w1 = int(np.abs(wbar * q1 / scale))

			max_a = np.argmax(self.q[st,:])

			a0, a1 = (0,0)
			if st == self.state:
				if max_a == 0:
					a0 = 255
					a1 = 0
				else:
					a0 = 0
					a1 = 255

			r0, g0, b0 = (0, 255, a0)
			r1, g1, b1 = (0, 255, a1)

			if q0 < 0:
				r0 = 255
				g0 = 0
			elif q1 < 0:
				r1 = 255
				g1 = 0

			c0 = (r0, g0, a0)
			c1 = (r1, g1, a1)

			wsum = np.sum(np.abs(self.q[st, :]))

			wm = int(wbar * max_a)
			ws = max(1, int(wbar * self.q[st, 1]))

			rect0 = Rect((x+wbar, y), (-w0, hbar))
			rect1 = Rect((2+x+wbar, y), (w1, hbar))

			pygame.draw.rect(self.board.surf, c0, rect0)
			pygame.draw.rect(self.board.surf, c1, rect1)

			#pygame.draw.rect(self.board.surf, (30,200,200), rect)
			#rect = Rect((x, y), (ws, hbar))
			#pygame.draw.rect(self.board.surf, color, rect)

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
		color = (200,0,0)
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

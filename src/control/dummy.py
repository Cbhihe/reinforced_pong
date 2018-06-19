from control.base import Controller

import pygame
from pygame import Rect
import numpy as np

class Follower(Controller):

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

class Keyboard(Controller):

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

class BallPredictor(Controller):

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

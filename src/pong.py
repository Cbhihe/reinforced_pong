# Encoding: utf-8

import pygame
from pygame.sprite import Sprite, collide_rect
from pygame import Rect,Surface

from control import *
import numpy as np

BOARD_MARGIN_X = 20
PADDLE_MARGIN_X = 50
PADDLE_MARGIN_Y = 20
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100
PADDLE_RADIUS = 200
PADDLE_TOP_SPEED = 3
BALL_DIAMETER = 10
BALL_INIT_SPEED = 5
SCORE_MARGIN_X = 150
SCORE_MARGIN_Y = 50

class Board(Sprite):
	def __init__(self, screen, size, cl, cr):
		Sprite.__init__(self)
		self.screen = screen
		self.size = size
		self.surf = Surface(size)
		self.rect = self.surf.get_rect()
	
		self.pl = Paddle(self, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'left')
		self.pr = Paddle(self, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'right')

		self.ball = Ball(self, BALL_DIAMETER)

		self.cl = cl(self, self.pl, self.ball)
		self.cr = cr(self, self.pr, self.ball)

		self.marker = {'left':0, 'right':0}
		self.restart = False
		self.player_scored = None

		# TODO: Use a good font
		self.font = pygame.font.SysFont(None, 72)
		self.run = True

	def draw_net(self):
		rect = self.surf.get_rect()
		top = rect.midtop
		bottom = rect.midbottom
		pygame.draw.line(self.surf, (150,150,150), top, bottom, 1)

	def draw_score(self):
		g = 200
		color = (g,g,g)
		ml = str(self.marker['left'])
		mr = str(self.marker['right'])
		surf_l = self.font.render(ml, True, color)
		surf_r = self.font.render(mr, True, color)

		rect_l = surf_l.get_rect()
		rect_r = surf_r.get_rect()

		# Place text in the middle of each player side
		cx = self.rect.centerx
		rect_l.centerx = cx/2
		rect_r.centerx = self.rect.right - cx/2

		rect_l.centery = SCORE_MARGIN_Y
		rect_r.centery = SCORE_MARGIN_Y

		self.surf.blit(surf_l, rect_l)
		self.surf.blit(surf_r, rect_r)

	
	def update(self, pause):
		self.events = pygame.event.get()
		for event in self.events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_q:
					#pygame.quit()
					#exit()
					self.run = False
				if event.key == pygame.K_r:
					# Restart counters
					self.marker = {'left':0, 'right':0}

		if self.marker['left'] >= 100 or self.marker['right'] >= 100:
			#print(self.marker)
			self.marker = {'left':0, 'right':0}

		if pause: return

		self.ball.update()

		self.cl.update()
		self.cr.update()

		self.pl.update()
		self.pr.update()

		if self.restart:
			self.ball.restart()
			self.restart = False
			self.player_scored = None
			#print(self.marker)

		#if self.marker['left'] >= 1000 or self.marker['right'] >= 1000:
		#	print(self.marker)
		#	self.run = False

	def draw(self):
		# Clear the board
		self.surf.fill((0,0,0))

		# Draw elements
		self.draw_net()
		self.draw_score()

		self.pl.draw()
		self.pr.draw()

		self.ball.draw()
		self.cl.draw()
		self.cr.draw()


		# Finally blit into the screen
		self.screen.blit(self.surf, dest=(0,0))

	def score(self, player):
		self.marker[player] += 1
		self.restart = True
		self.player_scored = player

	def get_ball_position(self, controller):
		if controller == self.cl:
			return self.ball.position

		# Mirror table
		bx, by = self.ball.position
		w, h = self.size

		bx = w - bx - 1
		
		return np.array([bx, by])

	def get_ball_speed(self, controller):
		if controller == self.cl:
			return self.ball.speed

		# Mirror ball speed
		vx, vy = self.ball.speed

		return np.array([-vx, vy])

	def get_ball_top_speed(self, controller):
		return self.ball.top_speed

	def get_player_position(self, controller, me=True):
		if me:
			if controller == self.cl:
				return self.pl.y
			else:
				return self.pr.y
		else:
			if controller == self.cl:
				return self.pr.y
			else:
				return self.pl.y

	def set_player_speed(self, controller, speed):
		if controller == self.cl:
			self.pl.set_speed(speed)
		else:
			self.pr.set_speed(speed)

	def get_player_speed(self, controller):
		if controller == self.cl:
			return self.pl.vy
		else:
			return self.pr.vy

	def get_debug_zone(self, controller):
		w,h = self.size
		rect = Rect((0, PADDLE_MARGIN_Y), (PADDLE_MARGIN_X, h-2*PADDLE_MARGIN_Y))

		if controller == self.cl:
			rect.right = PADDLE_MARGIN_X
			return rect
			
		else:
			rect.left = w - PADDLE_MARGIN_X
			return rect

	def get_paddle_rect(self, controller):
		if controller == self.cl:
			return self.pl.rect

		# Mirror paddle rect
		size = (PADDLE_WIDTH, PADDLE_HEIGHT)
		topleft = (PADDLE_MARGIN_X, PADDLE_MARGIN_Y)
		rect = Rect(topleft, size)
		rect.top = self.pr.rect.top
		
		return rect

	def get_marker(self, controller):
		ml = self.marker['left']
		mr = self.marker['right']

		if controller == self.cl:
			return {'me':ml, 'opponent':mr}
		else:
			return {'me':mr, 'opponent':ml}

	def get_player_scored(self, controller):
		side = self.player_scored

		if not side: return None

		if controller == self.cl:
			if side == 'left':
				return 'me'
			else:
				return 'opponent'

		else:
			if side == 'right':
				return 'me'
			else:
				return 'opponent'

	def get_paddle_top_speed(self, controller):
		return PADDLE_TOP_SPEED

	def get_paddle_status(self, controller):
		if controller == self.cl:
			return self.pl.status
		else:
			return self.pr.status
		
			
class Paddle(Sprite):
	def __init__(self, board, hmargin, vmargin, side):
		Sprite.__init__(self)
		self.board = board
		self.board_size = board.surf.get_size()
		self.margin = (hmargin, vmargin)

		self.size = (PADDLE_WIDTH, PADDLE_HEIGHT)
		self.rect = Rect((0, 0), self.size)

		# 1-pixel wide collision rect placed at the center side of the paddle
		self.crect = Rect((0, 0), (1, PADDLE_HEIGHT))
		w,h = self.board_size

		# Place the paddle at the top
		if side == 'left':
			self.rect.topleft = (hmargin, vmargin)
			self.crect.topright = self.rect.topright
		else:
			self.rect.topright = (w - hmargin, vmargin)
			self.crect.topleft = self.rect.topleft

		wboard, hboard = self.board_size
		wsurface = PADDLE_WIDTH
		hsurface = hboard - vmargin*2
		surf_size = (wsurface, hsurface)
		self.surf = Rect(self.rect.topleft, surf_size)

		self.side = side

		# The center y position of the paddle in floating point
		self.y = self.rect.centery

		# Paddle vertical speed
		self.vy = 0.0

		self.top_speed = PADDLE_TOP_SPEED
		self.status = None

	def set_speed(self, vy):
		if vy > self.top_speed:
			vy = self.top_speed
		if vy < -self.top_speed:
			vy = -top_speed

		self.vy = vy

	def set_position(self, y):
		rect = self.rect

		rect.centery = y
		top = self.surf.top
		bottom = self.surf.bottom

		if rect.top < top:
			rect.top = top
			y = rect.centery
		if rect.bottom > bottom:
			rect.bottom = bottom
			y = rect.centery

		# Place also the crect
		if self.side == 'left':
			self.crect.topright = self.rect.topright
		else:
			self.crect.topleft = self.rect.topleft

		self.y = y

	def status_ball(self, ball):
		ball_rect = ball.rect
		bx, by = ball_rect.center
		px, py = self.crect.center

		if self.crect.colliderect(ball_rect):
			self.status = 'collision'
		elif self.side == 'left' and bx < px:
			self.status = 'lost'
		elif self.side == 'right' and bx > px:
			self.status = 'lost'
		else:
			self.status = None

	def update(self):
		self.set_position(self.y + self.vy)

	def draw(self):
		# Draw the paddle in the rect position
		g = 200
		self.board.surf.fill((g,g,g), self.rect)
		#self.board.surf.fill((255,127,127), self.crect)


class Ball(Sprite):
	def __init__(self, board, diameter):
		Sprite.__init__(self)
		self.board = board
		self.diameter = diameter
		self.size = (diameter, diameter)
		self.ball = Surface(self.size)
		self.ball.fill((0,255,0))
		self.rect = Rect(self.board.rect.center, self.size)

		w,h = self.board.surf.get_size()

		# The ball center position in floating point
		self.position = (w/2, h/2)

		# Speed is measured in pixels / frame
		self.speed = np.array([3.0, 3.3])

		self.top_speed = diameter
		self.init_speed = BALL_INIT_SPEED

	def check_miss(self):
		# Test if the ball was missed by any paddle

		x, y = self.rect.center
		pr = self.board.pr

		pl = self.board.pl

		pl.status_ball(self)
		pr.status_ball(self)

		if pl.status == 'lost':
			self.board.score('right')
		elif pr.status == 'lost':
			self.board.score('left')

		
	def restart(self):
		# Put the ball in the center again
		self.rect.center = self.board.rect.center
		self.position = self.rect.center

		side = self.board.player_scored
		bw, bh = self.board.size

		# Select a random angle such that the ball goes directly to the paddle
		# area (margins are ignored?)
		max_slope = (bh/2) / (bw/2)
		max_angle = np.arctan(max_slope)
		angle = np.random.uniform(-max_angle, max_angle)

		# If the winner was the right one, the ball should move to the left
		if side == 'right':
			angle = np.pi + angle


		# Now we have the angle and the vector length
		phi = angle
		rho = self.init_speed
		vx = rho * np.cos(phi)
		vy = rho * np.sin(phi)

		speed = np.array([vx, vy])
		
		self.set_speed(speed)

	def collide_paddles(self):
		pr = self.board.pr
		pl = self.board.pl

		v = self.speed
		vx, vy = v

		#if collide_rect(self, pr) or collide_rect(self, pl):
		if collide_rect(self, pl) and vx < 0:

			cx, cy = pl.rect.center
			cx -= PADDLE_RADIUS

		elif collide_rect(self, pr) and vx > 0:

			cx, cy = pr.rect.center
			cx += PADDLE_RADIUS

		else:
			return

		bx, by = self.rect.center

		nx = bx - cx
		ny = by - cy

		n = np.array([nx, ny])

		# The new axis of reflectios for the trajectory
		n = n / np.linalg.norm(n)

		v = v - 2 * np.dot(v, n) * n


		# Avoid the ball to move close to 90 and 270 degrees (pi/2 and 3 pi/2)

		#angle_limit = 0.2 # About 11.5 degrees
		#if abs(angle - pi/2) < 0.2:
		#	if angle < pi/2:
		#		angle = pi/2 - angle_limit
		#	else:
		#		angle = pi/2 + angle_limit

		#if abs(angle - 3*pi/2) < 0.2:
		#	if angle < 3*pi/2:
		#		angle = 3*pi/2 - angle_limit
		#	else:
		#		angle = 3*pi/2 + angle_limit

		#vx = -vx
		
		# Speed up a bit the ball
		v *= 1.1

		self.set_speed(v)

	def set_speed(self, speed):
		speed = np.array(speed)

		abs_speed = np.linalg.norm(speed)

		# Avoid the ball from exceeding the top speed
		if abs_speed > self.top_speed:
			speed/abs_speed 
			speed *= self.top_speed/abs_speed
			#print('Ball limit speed reached')

		vx,vy = speed
		if np.abs(vx) < 1.0:
			vx = np.sign(vx) * 1.0

		speed = np.array([vx, vy])

		self.speed = speed

	def update(self):
		w,h = self.board.surf.get_size()

		self.check_miss()
		self.collide_paddles()

		px, py = self.position
		vx, vy = self.speed
		px += vx
		py += vy

		if px > w or px < 0:
			vx = -vx

		if py > h or py < 0:
			vy = -vy

		self.speed = (vx, vy)
		self.position = (px, py)

		self.rect.center = (px, py)

	def draw(self):
		self.board.surf.blit(self.ball, self.rect)


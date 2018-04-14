# Encoding: utf-8

import pygame, os
from pygame.sprite import Sprite
from pygame import Rect,Surface

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

BOARD_MARGIN_X = 20

PADDLE_MARGIN_X = 50
PADDLE_MARGIN_Y = 30

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100

class Board(Sprite):
	def __init__(self, screen, size):
		Sprite.__init__(self)
		self.screen = screen
		self.size = size
		self.board = Surface(self.size)
		self.init_paddles()
	
	def init_paddles(self):
		w,h = self.size

		self.pl = Paddle(self.board, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'left')
		self.pr = Paddle(self.board, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'right')
	
	def update(self):
		self.pl.update()
		self.pr.update()
		self.screen.blit(self.board, dest=(0,0))

class Paddle(Sprite):
	def __init__(self, board, hmargin, vmargin, side):
		Sprite.__init__(self)
		self.board = board
		self.board_size = board.get_size()
		self.paddle_size = (PADDLE_WIDTH, PADDLE_HEIGHT)

		wboard, hboard = self.board_size
		wsurface = PADDLE_WIDTH
		hsurface = hboard - vmargin*2

		self.surf_size = (wsurface, hsurface)

		# Position is a float from [0,1] with 0 the top, and 1 the bottom.
		self.position = 0.0

		self.surf = Surface(self.surf_size)
		self.surf_rect = Rect((0, 0), self.surf_size)
		w,h = self.board_size

		if side == 'left':
			self.surf_rect.topleft = (hmargin, vmargin)
		else:
			self.surf_rect.topright = (w - hmargin, vmargin)

		paddle_size = (PADDLE_WIDTH, PADDLE_HEIGHT)
		self.paddle = Surface(paddle_size)
		self.paddle.fill((255,255,255))
		self.paddle_rect = Rect((0, 0), paddle_size)

		self.side = side
		if side == 'left':
			self.x = 0
		else:
			self.x = w

	def update(self):
		incr = 0.02
		if self.side == 'left':
			self.position += incr
		else:
			self.position -= incr * 0.7

		if self.position > 1.0:
			self.position = 0.0
		elif self.position < 0.0:
			self.position = 1.0

		sw,sh = self.surf_size
		pw,ph = self.paddle_size

		x = 0
		y = self.position * (sh - ph)
		self.paddle_rect.topleft = (x, y)

		# Clear previos paddle
		self.surf.fill((100,0,0))

		# Place the paddle into the paddle surface, in the correct position
		self.surf.blit(self.paddle, self.paddle_rect)

		# Then place the updated paddle surface into the board
		self.board.blit(self.surf, self.surf_rect)


pygame.init()
screen = pygame.display.set_mode((320*2, 240*2))

size = (320*2, 240*2)
b = Board(screen, size)
clock = pygame.time.Clock()

while True:
	b.update()
	pygame.display.update()
	pygame.display.flip()

	clock.tick(60)

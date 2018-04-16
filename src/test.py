# Encoding: utf-8

import pygame, os, time
from pygame.sprite import Sprite, collide_rect
from pygame import Rect,Surface

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

BOARD_MARGIN_X = 20

PADDLE_MARGIN_X = 50
PADDLE_MARGIN_Y = 10

PADDLE_WIDTH = 10
PADDLE_HEIGHT = 100

BALL_DIAMETER = 10

DRAW = True
FPS = 60

class Board(Sprite):
	def __init__(self, screen, size):
		Sprite.__init__(self)
		self.screen = screen
		self.surf = Surface(size)
		self.rect = self.surf.get_rect()
	
		self.pl = Paddle(self.surf, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'left')
		self.pr = Paddle(self.surf, PADDLE_MARGIN_X, PADDLE_MARGIN_Y, 'right')

		self.ball = Ball(self, BALL_DIAMETER)

		self.cl = PaddleControllerFollower(self, self.pl, self.ball)
		self.cr = PaddleControllerKeyboard(self, self.pr, self.ball)

	def draw_net(self):
		rect = self.surf.get_rect()
		top = rect.midtop
		bottom = rect.midbottom
		pygame.draw.line(self.surf, (150,150,150), top, bottom, 1)
	
	def update(self):
		self.ball.update()

		self.cl.update()
		self.cr.update()

		self.pl.update()
		self.pr.update()

	def draw(self):
		# Clear the board
		self.surf.fill((0,0,0))

		# Draw elements
		self.draw_net()
		self.ball.draw()

		self.pl.draw()
		self.pr.draw()

		# Finally blit into the screen
		self.screen.blit(self.surf, dest=(0,0))


class PaddleControllerFollower:
	def __init__(self, board, paddle, ball):
		self.board = board
		self.paddle = paddle
		self.ball = ball

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

class PaddleControllerKeyboard:
	def __init__(self, board, paddle, ball):
		self.board = board
		self.paddle = paddle
		self.ball = ball

	def update(self):

		vy = self.paddle.vy
		events = pygame.event.get()
		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_UP:
					vy = -self.paddle.top_speed
				if event.key == pygame.K_DOWN:
					vy = self.paddle.top_speed
			elif event.type == pygame.KEYUP:
				vy = 0
	
		self.paddle.set_speed(vy)


class Paddle(Sprite):
	def __init__(self, board, hmargin, vmargin, side):
		Sprite.__init__(self)
		self.board = board
		self.board_size = board.get_size()
		self.margin = (hmargin, vmargin)

		self.paddle_size = (PADDLE_WIDTH, PADDLE_HEIGHT)
		self.rect = Rect((0, 0), self.paddle_size)
		w,h = self.board_size

		# Place the paddle at the top
		if side == 'left':
			self.rect.topleft = (hmargin, vmargin)
		else:
			self.rect.topright = (w - hmargin, vmargin)

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

		self.top_speed = 3.0

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

		self.y = y

	def update(self):
		self.set_position(self.y + self.vy)

	def draw(self):
		# Draw the paddle in the rect position
		self.board.fill((255,255,255), self.rect)


class Ball(Sprite):
	def __init__(self, board, diameter):
		Sprite.__init__(self)
		self.board = board
		self.diameter = diameter
		self.size = (diameter, diameter)
		self.ball = Surface(self.size)
		self.ball.fill((0,255,0))
		self.rect = Rect((0, 0), self.size)

		w,h = self.board.surf.get_size()

		# The ball center position in floating point
		self.position = (w/2, h/2)

		# Speed is measured in pixels / frame
		self.speed = (3.0, 3.3)

	def check_miss(self):
		# Test if the ball was missed by any paddle

		x, y = self.rect.center

		pr = self.board.pr



	def collide_paddles(self):
		pr = self.board.pr
		pl = self.board.pl

		vx, vy = self.speed

		if collide_rect(self, pr) or collide_rect(self, pl):
			vx = -vx

		self.speed = (vx, vy)


	def update(self):
		w,h = self.board.surf.get_size()

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


pygame.init()
screen = pygame.display.set_mode((320*2, 240*2))

size = (320*2, 240*2)
b = Board(screen, size)
clock = pygame.time.Clock()

frame = 1
delay = 10000

tic = time.time()
while True:
	b.update()

	if DRAW:
		b.draw()
		pygame.display.update()
		pygame.display.flip()
		clock.tick(FPS)
		delay = 200

	if not frame % delay:
		toc = time.time()
		fps = frame/(toc-tic)
		#print("FPS = {:.2f}".format(clock.get_fps()))
		print("Real fps = {:.2f}".format(fps))

	frame += 1

	

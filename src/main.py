from pong import *
from control import *
import pygame, os, time

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

SCREEN_SIZE = (640, 480)
DRAW = True
FPS = 60

def main():

	pygame.init()
	screen = pygame.display.set_mode(SCREEN_SIZE)

	cl = PCSimpleLearning
	#cl = PCFollower
	#cr = PCSimpleLearning
	cr = PCKeyboard

	b = Board(screen, SCREEN_SIZE, cl, cr)
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

if __name__ == '__main__':
	main()

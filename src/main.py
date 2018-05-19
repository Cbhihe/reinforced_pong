from pong import *
from control import *
import pygame, os, time

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

SCREEN_SIZE = (640, 480)
DRAW = True
#FPS = 20
FPS = 200

def main():

	print("Press f to toggle drawing (Fast mode)")

	pygame.font.init()
	pygame.display.init()
	pygame.display.set_caption('Skynet')

	screen = pygame.display.set_mode(SCREEN_SIZE)

	#cl = PCKeyboard
	cl = PC2
	#cl = PCFollower
	#cl = PC2
	#cl = PCFollower
	#cr = PC2
	#cr = PCKeyboard
	cr = PCFollower
	#cr = PCPredictor
	#cr = PCPredictorLearn

	b = Board(screen, SCREEN_SIZE, cl, cr)
	clock = pygame.time.Clock()

	frame = 1
	delay = 10000
	pause = False

	draw = DRAW
	tic = time.time()
	while b.run:

		b.update(pause)

		events = b.events


		for event in events:
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_f:
					#pygame.quit()
					#exit()
					draw = not draw
				elif event.key == pygame.K_p:
					pause = not pause

		if draw or (not frame % delay):
			b.draw()
			pygame.display.update()
			pygame.display.flip()
			clock.tick(FPS)
			delay = 200
		else:
			delay = 10000


		if not frame % delay and not draw:
			toc = time.time()
			fps = frame/(toc-tic)
			#print("FPS = {:.2f}".format(clock.get_fps()))
			#print(b.marker)
			#print("Real fps = {:.2f}".format(fps))
			tic = time.time()
			frame = 1

		frame += 1

if __name__ == '__main__':
	main()

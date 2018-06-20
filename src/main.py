from pong import *

import control
import pygame, os, time, pickle
import os.path
import gzip

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

SCREEN_SIZE = (640, 480)
DRAW = True
#FPS = 20
FPS = 200
TRAIN_DIR = 'train'
TRAINING = True
#TRAIN_TIME = 30 * 60 # 30 min for training
TRAIN_TIME = 30

# The controller in use
# TODO: Select from cmdline
CONTROLLER = control.QLd1()


def get_train_path(controller):
	name = controller.__class__.__name__
	fpath = os.path.join(TRAIN_DIR, name + '.data.gz')
	return fpath

def load_controller(controller):
	fpath = get_train_path(controller)
	data = restore(fpath)
	controller.restore(data)

def tournament():
	controllers = [CONTROLLER]
	name = [c.__class__.__name__ for c in controllers]

	# Reference controller for training
	ref = control.Follower()
	ref_name = ref.__class__.__name__

	t = TRAIN_TIME

	for c in controllers:
		name = c.__class__.__name__
		print('Training {}'.format(name))
		b = Board(None, SCREEN_SIZE, c, ref, display=False)
		cdata, _ = train(b, c, ref, t)

		savepath = os.path.join(TRAIN_DIR, name + '.data.gz')
		save(cdata, savepath)

		print("Trained data saved in {}".format(savepath))

def measure(left, right, maxmatches=20):
	"""A trained controller is played against a reference controller, and
	performance is measured"""

	b = Board(None, SCREEN_SIZE, left, right, display=False)
	
	frame = 1
	delay = 10000
	tic = time.time()

	matches = -1

	while b.matches < maxmatches:
		if matches != b.matches:
			matches = b.matches
			print("Match {} of {}. Status: {}".format(
				b.matches, maxmatches, b.win_matches))
		b.update()
		frame += 1

	wins = b.win_matches
	print(wins)


def play(screen, controller, trainfile=None):

	name = controller.__class__.__name__
	if trainfile != None:
		data = restore(trainfile)
		controller.restore(data)

	# Reference controller for playing
	ref = control.Follower()
	ref_name = ref.__class__.__name__

	b = Board(screen, SCREEN_SIZE, controller, ref)
	
	clock = pygame.time.Clock()

	frame = 1
	delay = 10000
	pause = False

	draw = DRAW
	tic = time.time()

	print("Press f to toggle (f)ast drawing")
	print("Press d to toggle (d)ebug drawings")
	print("Press p to (p)ause")
	print("Press q to (q)uit")

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
	

def train(b, left, right, t):
	"Train 2 agents by playing the game t seconds"


	frame = 1

	tic = time.time()
	toc = tic + t
	while time.time() < toc:
		b.update(False)
		frame += 1

	print('Trained for {:.2f} s with {} frames and {} matches'.format(
		time.time() - tic, frame, b.matches))

	ldata = left.save()
	rdata = right.save()

	return (ldata, rdata)

def save(data, fpath):
	with gzip.open(fpath, 'wb') as f:
		pickle.dump(data, f)

def restore(fpath):
	with gzip.open(fpath, 'rb') as f:
		data = pickle.load(f)
	
	return data

def main(training=False):

	#left = control.QL2()
	#right = control.Follower()

	## Load trained controller
	#load_controller(left)

	#measure(left, right)
	#return

	if training:
		tournament()
	else:
		pygame.font.init()
		pygame.display.init()
		pygame.display.set_caption('skynet')
		screen = pygame.display.set_mode(SCREEN_SIZE)

		# Play the only trained controller
		c = CONTROLLER
		#trainfile = get_train_path(c)
		trainfile = None
		play(screen, c, trainfile)


if __name__ == '__main__':
	main(TRAINING)

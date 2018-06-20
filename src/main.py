from pong import *

import control
import pygame, os, time, pickle, sys
import os.path
import gzip
from plot import plot_all
import argparse

# Center window
os.environ['SDL_VIDEO_CENTERED'] = '1'

SCREEN_SIZE = (640, 480)
DRAW = True
FPS = 200


# Default configuration: training mode for 30 min
TRAIN_DIR = 'train'
TRAINING = True
TRAIN_TIME = 30 * 60 # 30 min for training


# The controller in use
parser = argparse.ArgumentParser(description='Play or train a Pong game controller')

parser.add_argument("-c", "--controller",
	help="the name of the controller to use", type=str, required=True)

parser.add_argument("-p", "--play", help="play the trained controller",
	action="store_true")

parser.add_argument("-t", "--time", help="time in seconds for training",
	type=int, default=TRAIN_TIME)

parser.add_argument("-f", "--file", help="file used to restore training data",
	type=str)

args = parser.parse_args()

if args.play:
	TRAINING = False

TRAIN_TIME = args.time

try:
	CONTROLLER = globals()[args.controller]()
except KeyError:
	print("Controller '{}' not found".format(args.controller))
	exit(1)


def get_train_path(controller, train_time):
	name = controller.__class__.__name__
	train_time = str(train_time)
	fpath = os.path.join(TRAIN_DIR, name, train_time, 'data.gz')
	return fpath


if args.file != None:
	TRAIN_FPATH = args.file
else:
	TRAIN_FPATH = get_train_path(args.controller, args.time)



if TRAINING:
	print('Controller selected {}, train time {} seconds'.format(
		args.controller, TRAIN_TIME))
else:
	print('Controller selected {}, play mode'.format(
		args.controller))



def load_controller(controller, train_time):
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
		print('Training {}'.format(name), file=sys.stderr)
		b = Board(None, SCREEN_SIZE, c, ref, display=False)

		train_dir = '{}/{}/{}'.format(TRAIN_DIR, name, t)

		if not os.path.exists(train_dir):
		    os.makedirs(train_dir)

		csv_file = os.path.join(train_dir, 'train.csv')

		with open(csv_file, 'w') as csvfile:
			c.log_file = csvfile
			cdata, _ = train(b, c, ref, t)

		savepath = os.path.join(train_dir, 'data.gz')
		save(cdata, savepath)

		print("Trained data saved in {}".format(savepath), file=sys.stderr)

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
		time.time() - tic, frame, b.matches), file=sys.stderr)

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

def main():

	#left = control.QL2()
	#right = control.Follower()

	## Load trained controller
	#load_controller(left)

	#measure(left, right)
	#return

	if TRAINING:
		tournament()
	else:
		pygame.font.init()
		pygame.display.init()
		pygame.display.set_caption('Controller {}'.format(args.controller))
		screen = pygame.display.set_mode(SCREEN_SIZE)

		# Play the only trained controller
		c = CONTROLLER
		#trainfile = get_train_path(c)
		play(screen, c, TRAIN_FPATH)


if __name__ == '__main__':
	main()
	if TRAINING:
		plot_all()

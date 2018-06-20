## Running

To run the simulator, first you need to select which controller you want to
train. At the time of writing, the following controllers are available:

	hop% grep class control/param.py 
	class QL1(QL):
	class QL2(QL):
	class QLd1(QLd):
	class QLd2(QLd):
	class QLe1(QLe):
	class QLe2(QLe):
	class QLe3(QLe2):
	class SARSA1(SARSA):
	class SARSA2(SARSA):
	class DQN1(DQN):

But it is expected for them to grow, as we add more parametriced versions. For
example let's take `SARSA1`.

Next, we run the simulator with the chosen controller:

	hop% python main.py -c SARSA1

This process will last 30 min, the default time for training. Check the `--help`
option, for how to specify other time limit:

	hop% python main.py --help    
	usage: main.py [-h] -c CONTROLLER [-p] [-k] [-t TIME] [-f FILE]

	Play or train a Pong game controller

	optional arguments:
	  -h, --help            show this help message and exit
	  -c CONTROLLER, --controller CONTROLLER
				the name of the controller to use
	  -p, --play            play the trained controller
	  -k, --keyboard        use the keyboard when playing
	  -t TIME, --time TIME  time in seconds for training
	  -f FILE, --file FILE  file used to restore training data

The trained state of the controller will be saved by default in
`train/SARSA1/1800/data.gz`.

If you want to select a trained controller and play the game visualy, you need
the option `-p`. Use `-k` also to enable the keyboard and play against the machine:

	hop% python main.py -p -k -c QL2

## Dependencies

The following dependences are needed for python 3:

	python-keras-preprocessing
	python-keras-applications
	python-keras
	python-tensorflow
	python-numpy
	python-matplotlib



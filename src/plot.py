import matplotlib.pyplot as plt
import numpy as np
import os

TRAINDIR='train'
FIGDIR='fig'


def plot_group_point_diff():
	sarsa_controllers = ['QL1', 'QL2', 'QLe2', 'SARSA1', 'SARSA2', 'DQN1']
	train_time = str(30*60)

	for controller in sarsa_controllers:
		fpath = os.path.join(TRAINDIR, controller, train_time, "train.csv")

		data = np.genfromtxt(fpath, skip_header=1, comments="#", delimiter=" ")

		t = data[:,0]
		iteration = data[:,1]
		mean_reward = data[:,4]
		points_me = data[:,9]
		points_opp = data[:,10]

		denom = 1 + points_me + points_opp
		point_diff = (points_me - points_opp)/denom

		plt.plot(iteration[20:], point_diff[20:], label=controller)

	plt.title('With {} seconds of training'.format(train_time))
	plt.xlabel('Iteration')
	plt.ylabel('Normalized score difference')
	plt.legend()
	plt.grid()

	fig_fpath = os.path.join(FIGDIR, 'all-norm-score-diff.png')
	plt.savefig(fig_fpath)
	plt.close()

		


def plot(controller, train_time, fpath):

	fig_dir = os.path.join(FIGDIR, controller, train_time)

	if not os.path.exists(fig_dir):
		os.makedirs(fig_dir)

	train_time = str(train_time)

	data = np.genfromtxt(fpath, skip_header=1, comments="#", delimiter=" ")
	t = data[:,0]
	iteration = data[:,1]
	mean_reward = data[:,4]
	points_me = data[:,9]
	points_opp = data[:,10]

	denom = 1 + points_me + points_opp
	point_diff = (points_me - points_opp)/denom

	plt.plot(t, mean_reward)
	plt.title('Controller {} trained for {} seconds'.format(controller,
		train_time))
	plt.xlabel('CPU time in seconds')
	plt.ylabel('Mean reward of the last 1000 iterations')
	fig_fpath = os.path.join(fig_dir, 'mean-reward-vs-cputime.png')
	plt.savefig(fig_fpath)
	plt.close()

	plt.plot(iteration[50:], point_diff[50:])
	plt.title('Controller {} trained for {} seconds'.format(controller,
		train_time))
	plt.xlabel('Iteration')
	plt.ylabel('Point difference from reference controller')
	fig_fpath = os.path.join(fig_dir, 'point-diff-vs-iteration.png')
	plt.savefig(fig_fpath)
	plt.close()

def plot_all():
	csv_files = []

	for root, dirs, files in os.walk(TRAINDIR):
		for fn in files:
			if fn.endswith('.csv'):
				csv_files.append(os.path.join(root, fn))
	data = []

	for fpath in csv_files:
		parts = fpath.split("/")
		controller = parts[1]
		train_time = parts[2]
		data.append((controller, train_time, fpath))

	for controller, train_time, fpath in data:
		print('Plotting {}'.format(fpath))
		plot(controller, train_time, fpath)


if __name__ == '__main__':
	print('Updating all plots with the training data from {}'.format(TRAINDIR))
	#plot_all()
	plot_group_point_diff()

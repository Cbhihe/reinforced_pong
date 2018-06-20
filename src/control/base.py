import sys,time

class Controller:
	def __init__(self):
		self.board = None

	def set_board(self, board):
		self.board = board

	def save(self):
		"Save the status of the controller"
		return None

	def restore(self, data):
		"Restore the status of the controller"
		pass

	def update(self):
		"Update the state of the paddle by looking at the game state"
		raise NotImplementedError()

	def draw(self):
		"Draw some debug information if neccesary"
		# DO NOT attempt to draw the paddle here!
		pass

class ControllerLog(Controller):
	def __init__(self):
		super().__init__()
		self.log_now = False
		self.log_interval = 1000 # Log only after these iterations
		self.log_file = sys.stdout
		self.iteration = 0
		self.header_printed = False
		self.start_time = time.clock()

	def log_header(self):
		raise NotImplementedError()

	def log(self):
		raise NotImplementedError()

	def update(self):
		if not self.header_printed:
			self.log_header()
			self.header_printed = True

		if (self.iteration % self.log_interval) == 0:
			self.log()

		self.iteration += 1


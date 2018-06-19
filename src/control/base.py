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


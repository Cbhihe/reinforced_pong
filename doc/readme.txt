Useful info + code: https://github.com/dennybritz/reinforcement-learning

First the game simulator was implemented, so we could see visually what was 
going on using pygame, and simple collision mechanisms (all objects considered 
boxes).

The game is structured in three main blocks. 
	- pong.py: game related elements, but no logic
	- control.py: all controllers (connectors between human and paddle,
		+ connectors between agent-controller
	- main.py: attribution of a side to each agent-controller.
		Note: must add cli parameters method instead of modifying code 
		by hand), e.g.: $ python main.py -l keyboard -r ql

Main loop
---------
The game performs the following actions by order, in a infinite loop:

	- Read events from keyboard and store in board
	- Update the ball
	- Update the left controller
	- Update the right controller
	- Update the left paddle
	- Update the right paddle
	- Restart the game if needed

Important: Both controllers must be updated before paddles are updated. If not
one controller could gain advantage from the movement of its opponent.

The ball implements some logic to determine when a player scores, and handles
the ball trajectory by colliding with paddles and board boundaries.

Controllers
-----------
Each controller implements the method update, and has access to the board
object, which provides information from the board, as well as the actions that
can be performed.

In our first code version, each controller’logic was programmed as a function 
of the placement of its paddle, on either the left or the right side of the board.
The consequence was a chirality problem, as a left-hand-side (lhs) trained
controller (a lefty) could perform erroneously when placed on the rhs of the pong
board, and viceversa. 
For that reason we modified the program’s design and let the board take control of
symmetry issues, in such a way that all controllers see the game as if they were
placed on an undifferentiated (and in our special case, left) side.

QL agent-controller (PCv1)
---------------------------
Translation of game status into a state, as needed by the state-action—reward
mapping paradigm of QL was performed as follows.

Our simplest approach consisted in determining whether the ball was placed above
or below the paddle’s center
	s0: above
	s1: below
To this we added two actions: 
	a0: move up
Finally the agents’ rewards were:
	agent scores: +1
	agent’s opponent scores: -1
	agent’s paddle catches ball: +0.1 	(optional)
The above rudimentary agent quickly (how quickly ???) learnt how to play the game.

Important note: When designing our algorithmic QL methods, the main loop calls  update only once per frame. So in order to access states at any step, states need to be stored in the controller, until the next update takes place.
Updates for the positions of the ball and of the two paddles are performed outside the controller.  Once those updates are complete, objects have the opportunity to interact with the new state, as recorded in the controller.


Advanced Q-Learning controller (PCv2)
------------------------------------
State information such as ball speed and position, paddles’ positions are further discretized. == and generate an integer number by computing the cartesian product.==
As the number of states grows, so does learning time.  However “Learned” agents are now more discriminatory.  They become capable of directing the ball they just hit toward specific areas of the board which their opponent cannot reach or can reach only with difficulty.

As we further increased the complexity, so much training time became necessary, that it becomes impractical.  To overcome that limitation, an alternative method to encode board information was DQN.  
DQN could reduce greatly the complexity by using some interpolation.

Goals of the project
--------------------
The main goals are:

1. To determine if a machine can learn to play the game of
Pong without more information that the one provided by playing the game itself.

2. To let different controllers learn to play and determine which one
scores more in a kind of tournament, and with which parameters.

3. To compare the ability of the machine with a human, and see if they can win.

As side goals, we should learn the details of different Reinforcement Learning
methods, an how to transform and solve a real time problem.


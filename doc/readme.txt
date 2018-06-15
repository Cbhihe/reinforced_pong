Useful info + code: https://github.com/dennybritz/reinforcement-learning

What we did so far
------------------

First the game simulator was implemented, so we could see visually what was going on
using pygame, and simple collision mechanisms (all objects considered boxes).

The game is structured in three main blocks. In pong.py we have all elements
related with the game itself, but without any logic. In control.py we place all
the controllers that act either as a conector between the human and the paddle,
or between an automatic controller. Finally, the main.py file just sets the
choosen controllers in each side (we should add some parameters instead of
modify the code by hand, like: python main.py -l keyboard -r ql)

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

It is important to update both controllers before the paddles are updated,
otherwise one controller could gain advantage from the movement of the adversary.

The ball implements some logic to determine when a player scores, and handles
the ball trajectory by colliding to the objects and bounds of the board.

Controllers
-----------

Each controller implements the method update, and has access to the board
object, which provides information from the board, as well as the actions that
can be performed.

At the first iteration, each controller was aware of the global placement of the
paddle, so we run into a symmetry problem. A left trained controller could
perform wrongly if was after placed in the right side, and viceversa. So we
decided to let the board take control of the side, in a way that all controllers
see the game as if they were placed on the *left* side. Now each pair of trained
controllers can play, without any problem of the side.

Q-Learning controller (PC1)
---------------------------

In order to translate the status of the game into a state required by QL, we use
some approaches. The simplest one, is to determine if the ball is above or below
the paddle center, with states s0 and s1 respectively. Then the two actions,
move up, move down as a0 and a1. For the rewards, we have +1 if the player
scores, -1 if the adversary scores and an optional +0.1 (or similar) if the
paddle collides with the ball.

This simple controller quickly learns how to play the game, but the strategies
that could made use of are very limited

A important note should be taken into account when designing QL methods. The
main loop calls update only once per frame. So in order to access the state at t
and t+1, we need to store them in the controller, until the next update takes
place. Otherwise, if the update is performed inside the controller, the other
object don't have the opportunity to interact with the new state of the paddle.

Advanced Q-Learning controller (PC2)
------------------------------------

In order to provide useful information such as the ball speed, position and paddles
positions, we can discretize the real values that can take, and generate an
integer number by computing the cartesian product.

This method takes longer to train, as the number of states grow exponentially as
more discretizations are added. However, it can use some strategies, like
throwing the ball through the small margins, so the opponent cannot reach it.

As we increase the complexity this method takes so much time to train that we
cannot use it after some limits. In order to solve thi exponential growth, we
seek to alternative methods to encode the board information. One of the
proposals is DQN which could reduce greatly the complexity by using some
interpolation.

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


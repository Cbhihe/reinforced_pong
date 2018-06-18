## Experiments with and around Reinforced Learning

We study a closed system consisting of two autonomous and independent, temporally situated, learning agents, playing a game of Pong.  The game of ‘pong’ was one of the earliest video games, released in 1972 by Atari. It was built with simple 2D graphics.  

In the present Python implementation each-agent-player must overcome its opponent by learning to play the game better and faster in order to score points.  An agent is computationally autonomous in that it learns to interact with its environment by being rewarded, whenever its scores, and penalized whenever its opponent scores.  The goal-directed machine learning (ML) methods of choice in our case are reinforced learning (RL) methods, which we set out to implement and benchmark.  

Pong is a simple game and its rules are outlined at the end of this introductory section.  We choose to focus not on the implementation of the game [1], although it is far far from being devoid of interest, but rather on that of the ML methods we propose to study.  By endowing the two player-agents with different characteristics and learning method’s parameters, we set an explicit objective for them: to maximize their own score.  For that we make them aware of their environment in a manner detailed in the complete report.  Our goal is to compare the relative performances of different ML methods. 

Apart from the simplicity of the game, there are two ML-related main reasons to choose Pong to study the relative performance of different RL methods: 

1. Pong has two players.  It affords us the possibility to either pit one learning agent against another or to appraise an agent-player’s learning curve when opposed to a human player or to a training wall.  This eases the design of a parametric study of learning performance, as a function of learning methods’ parameters.  We can also allow two (differently configured) learning agents to compete in gradually more complex learning environments, i.e. environments with increasing numbers of actions and states.  
2. Given the nature of the problem, we study several RL methods [2], [3], in particular:
   * basic, off policy, model-free Q-Learning (QL), 
   * basic, on-policy, model-free State-Action-Reward-State-Action (SARSA),
   * Deep Q Neural-Networks (DQN)

#### **The simple game of ‘pong’**
The game consists of a rectangular arena (in the XY plane), a ball, and two paddles to hit the ball back and forth across the arena.  A player-agent (represented by a paddle) scores when the ball flies past the opposite player’s paddle and hits the back-wall opposite the scoring player’s side.  When this occurs a new episode, made of a sequence of exchanges, starts.
Each player can only move vertically (i.e. along direction Y).  The ball can bounce off the paddles as well as the side walls running parallel to axis X. The game is played in episodes. An episode’s terminal state occurs when one player scores against the other. 

--------------
### **References**:
**[1]**	   T. Appleton, “Writing Pong using Python and Pygame,” [blog post](https://trevorappleton.blogspot.com/2014/04/writing-pong-using-python-and-pygame.html) by Trevor Appleton, April 2014.  
**[2]**   R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction. Cambridge, MA, USA: MIT Press, 2018.  
**[3]**   C. Bishop, Pattern Recognition and Machine Learning. New York: Springer-Verlag, 2006.

# SnakeGame-IL181

This project seeks to implement the snake game using Q-Learning. 

# Instruction:

Install gym on your computer, and use this folder gym_snake to run the environment. The available gym_snake module from OpenAI gym runs too slow and has a rendering problem so I decided to create this different version that runs much faster without any problems. To change the size of the snake game, go to gym_snake/envs/snake_env and change the self.width and self.height parameters (default is (15,15)). You can also change the reward values, dafult is -1 reward each step, 100 if the snake eat the dot, -100 if it dies, and 10,000,000,000 if it wins the game.

File snake_bfs.py tries to solve the problem using Breadth First Search and performs relatively well, usually eating quite a few apples before dying, with a total reward of +300 on average (the file also plot out its reward each game). Its biggest problem is when the snake gets long enough, it tends to trap itself and lose the game, which is typically unavoidable at some point since -1 reward is applied each move (otherwise, we can create a pattern that goes through all square over and over, which guarantees winning the game, but the accumulated -1 rewards would be too high). 

As for Q-Learning in the snake_QLearning.py file, while it is theorectically possible to find an optimal strategy for this game using reinforcement learning, the possible number of states in the game is exponentially high as the snake can get to hundreds of tiles long, with countless positional variations for the snake. Thus, it is impractical to use this default state for Q-Learning. Instead, I consider the state as a vector of 6 variables ($a_1$,$a_2$,$a_3$,$a_4$,w,h) with the first 4 variables are booleans to see whether the immediate left, right, up, down of the snake head has an obstacle, encouraging it to avoid ones that have obstacles (wall or itself). The last two shows the width and height difference between the snake head and the dot, incentivizing the snake to approach the dot by minimizing these values. This strategy has a greedy nature, to minimize the distance between the snake and the dot and doesn't consider whether it might trap itself while doing so. Hence, it is very likely the optimal result will perform no better than its BFS counterpart. In exchange to reduce performance, the possible number of states are less than $2^4*15*15=3600$ states (depending on the board size). 

Hence, this Breadth First Search strategy is the threshold that I want the Q-learning algorithm to achieve. If the algorithm reach reward value near or beyond BFS strategy, Q Learning would then can be a useful tool for the snake game.  

# Results Analysis

Even with reduced number of states, it is still a quite large number, and hence would likely need thousands of game to converge. In my default parameters setting in snake_QLearning.py, the Q table has yet to converge, so the snake performs very poorly compared to its counterpart BFS. It is possible with enough games training, Q-Learning would eventually approach BFS performance, but will require a very high number of games, with each game costing roughly 5 seconds on average to play. I predict with $\alpha = \gamma = 0.1$ we would need more than $\frac{3600}{0.1} = 36000$ games to converge, which will take a lot of time to train, 50+ hours. Thus, I conclude it is impractical to apply Q-Learning in the snake game when a fast BFS algorithm already outperforms it.

import gym
import gym_snake
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import copy

from gym_snake.envs.snake_env import SnakeCellState
from gym_snake.envs.snake_env import SnakeReward
env = gym.make('snake-v0')

#Parameters
train_len = 2000
batch_size = 10
test_len = 5
epsilon = 0.1
alpha = 0.33
gamma = 0.33


Q_table = {}

#Train for a few batches, each of which will see how well the snake performs so far (to see if it improves over time)
plot = []
for i in range(batch_size):

    #Train phase:
    for i in range(train_len):
        count = 0
        next_state = []
        while True:
            square = env.game.head()
            state = []
            best = np.zeros(4)
            r = np.zeros(4)
            for k,i in enumerate([(square[0]-1,square[1]),(square[0]+1,square[1]),(square[0],square[1]+1),(square[0],square[1]-1)]):
                #Note that wall refers to both the boundary and the snake itself
                if env.game.cell_state(i) == SnakeCellState.WALL:
                    state.append(1)
                else:
                    state.append(0)

                #Future state
                new_state = []
                (s,t) = copy.deepcopy(i)
                for j in [(s-1,t),(s+1,t),(s,t+1),(s,t-1)]:
                    if env.game.cell_state(j) == SnakeCellState.WALL:
                        new_state.append(1)
                        r[k] = SnakeReward.DEAD
                    else:
                        new_state.append(0)
                        if env.game.cell_state(j) == SnakeCellState.DOT:
                            r[k] = SnakeReward.DOT
                        else:
                            r[k] = SnakeReward.ALIVE

                new_state.append(s-env.game.dot[0])
                new_state.append(t-env.game.dot[1])
                new_state = tuple(new_state)
                if new_state in Q_table:
                    if np.argmax(Q_table[new_state]) > best[k]:
                        best[k] = np.argmax(Q_table[new_state])
                else:
                    Q_table[new_state] = np.zeros(4)

            #Back to the present
            state.append(square[0]-env.game.dot[0])
            state.append(square[1]-env.game.dot[1])
            state = tuple(state)
            
            if state in Q_table:
                if random.random() > epsilon:
                    observation, reward, done, info = env.step(np.argmax(Q_table[state]))
                else:
                    observation, reward, done, info = env.step(random.randint(0,3))
            else:
                Q_table[state] = np.zeros(4)
                observation, reward, done, info = env.step(random.randint(0,3))

            #Q Learning Update
            Q_table[state] += alpha*(r+gamma*best-Q_table[state])

            count += 1
            if done or count > 250:
                break

    #Test phase
    total_r = 0
    for i in range(test_len):
        env.reset()
        r = 0
        count = 0
        while True:
            #env.render()
            square = env.game.head()
            state = []
            for i in ([(square[0]-1,square[1]),(square[0]+1,square[1]),(square[0],square[1]+1),(square[0],square[1]-1)]):
                #Note that wall refers to both the boundary and the snake itself
                if env.game.cell_state(i) == SnakeCellState.WALL:
                    state.append(1)
                else:
                    state.append(0)

            state.append(square[0]-env.game.dot[0])
            state.append(square[1]-env.game.dot[1])
            state = tuple(state)

            if state in Q_table:
                observation, reward, done, info = env.step(np.argmax(Q_table[state]))
            else:
                observation, reward, done, info = env.step(random.randint(0,3))
            
            r += reward
            count += 1
            #time.sleep(0.1)
            #termination
            if done or count > 250:
                print('episode {} finished with {} reward'.format(i,r))
                total_r += r
                break
    plot.append(total_r/test_len)

plt.plot(range(batch_size,batch_size*train_len+1,train_len),plot)
plt.xlabel('Training size')
plt.ylabel('Average Test Reward')
plt.show()

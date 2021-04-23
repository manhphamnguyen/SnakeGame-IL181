import gym
import gym_snake
import random
import time
from matplotlib import pyplot as plt
import numpy as np
import copy
import pickle
from sklearn.linear_model import LinearRegression

from gym_snake.envs.snake_env import SnakeCellState
from gym_snake.envs.snake_env import SnakeReward
env = gym.make('snake-v0')

#Parameters
batch_size = 1000
test_len = 5
epsilon = 0.1
alpha = 0.1
gamma = 0.1

#Load the q value table
file = open("q.pkl", "rb")
#Q_table = pickle.load(file)
#If you want to start from scratch
Q_table = {}

#Train for a few batches, each of which will see how well the snake performs so far (to see if it improves over time)
plot = []
for _ in range(batch_size):
    env.reset()

    #Train phase
    #First, we simulate the game
    train_reward = []
    next_state = []
    actions = []
    count = 0
    while True:
        square = env.game.head()
        state = []
        for k,i in enumerate([(square[0]-1,square[1]),(square[0]+1,square[1]),(square[0],square[1]+1),(square[0],square[1]-1)]):
            #Check if there's a wall on the 4 sides of the snake head (its body counts)
            #Note that wall refers to both the boundary and the snake itself
            if env.game.cell_state(i) == SnakeCellState.WALL:
                state.append(1)
            else:
                state.append(0)

        #Check the snake head's position compared to the dot, 1 if it's on the top or left, 0 if it matches, -1 if it's bottom or right.
        if square[0] > env.game.dot[0]:
            state.append(-1)
        elif square[0] == env.game.dot[0]:
            state.append(0)
        else:
            state.append(1)

        if square[1] > env.game.dot[1]:
            state.append(-1)
        elif square[1] == env.game.dot[1]:
            state.append(0)
        else:
            state.append(1)        

        state = tuple(state)
        
        #Choose an action based on the Q table and epsilon, make random moves once in a while and when Q-table is yet to reach the state
        if state in Q_table:
            if random.random() > epsilon:
                action = np.argmax(Q_table[state])
            else:
                action = random.randint(0,3)
        else:
            Q_table[state] = np.zeros(4)
            action = random.randint(0,3)

        #Reward for the action, encouraging the snake head to converge towards the apple
        if len(next_state) >= 1:
            reward = 0
            last_state = next_state[-1]
            if last_state[4] == 0 and state[4] != 0:
                reward -= 10
            if last_state[4] != 0 and state[4] == 0:
                reward += 10
            if last_state[5] == 0 and state[5] != 0:
                reward -= 10
            if last_state[5] != 0 and state[5] == 0:
                reward += 10
            if action == 0 and state[4] == -1:
                reward += 10
            if action == 1 and state[4] == 1:
                reward += 10
            if action == 3 and state[5] == -1:
                reward += 10
            if action == 2 and state[5] == 1:
                reward += 10

            train_reward.append(reward)
        actions.append(action)
        next_state.append(state)

        observation, reward, done, info = env.step(action)
        count += 1

        #Stop when termination condition reached (snake dies or eat the apple or too many moves)
        if done or count > 250:
            train_reward.append(reward)
            break

    #Now we train the Q-table from the data collected in that game
    train_reward = train_reward[::-1]
    next_state = next_state[::-1]
    actions = actions[::-1]
    for i in range(len(next_state)):
        #Q Learning Update
        if i == 0:
            Q_table[next_state[i]][actions[i]] = train_reward[i]
        else:
            Q_table[next_state[i]][actions[i]] += alpha*(train_reward[i]+gamma*max(Q_table[next_state[i-1]])-Q_table[next_state[i]][actions[i]])

    #Test phase
    total_r = 0
    for i in range(test_len):
        env.reset()
        r = 0
        count = 0
        while True:
            #if _ % 2000 == 0:
            #    env.render()
            #State creation, the same as train phase
            square = env.game.head()
            state = []
            for i in ([(square[0]-1,square[1]),(square[0]+1,square[1]),(square[0],square[1]+1),(square[0],square[1]-1)]):
                #Note that wall refers to both the boundary and the snake itself
                if env.game.cell_state(i) == SnakeCellState.WALL:
                    state.append(1)
                else:
                    state.append(0)

            if square[0] > env.game.dot[0]:
                state.append(-1)
            elif square[0] == env.game.dot[0]:
                state.append(0)
            else:
                state.append(1)

            if square[1] > env.game.dot[1]:
                state.append(-1)
            elif square[1] == env.game.dot[1]:
                state.append(0)
            else:
                state.append(1)
            state = tuple(state)

            #Choose an action, does not use epsilon nor update the Q_table
            if state in Q_table:
                observation, reward, done, info = env.step(np.argmax(Q_table[state]))
            else:
                observation, reward, done, info = env.step(random.randint(0,3))
            
            r += reward
            count += 1
            #time.sleep(0.1)
            #termination
            if done or count > 250:
                #print('episode {} finished with {} reward'.format(i,r))
                total_r += r
                break
    plot.append(total_r/test_len)

file = open("q.pkl", "wb")
pickle.dump(Q_table, file)
file.close()

model = LinearRegression()
model.fit(np.array(range(batch_size)).reshape(-1,1),np.array(plot).reshape(-1,1))

plt.scatter(range(batch_size),plot)
plt.plot([0,batch_size],[model.intercept_,model.coef_*batch_size+model.intercept_],color = 'red')
print(model.coef_*batch_size+model.intercept_)
print(plot[-1])

plt.xlabel('Training size')
plt.ylabel('Average Test Reward')
plt.show()

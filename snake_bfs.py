import gym
import gym_snake
import random
import time
import queue
import copy
import matplotlib.pyplot as plt

from gym_snake.envs.snake_env import SnakeCellState
env = gym.make('snake-v0')

#If there's a way from the head to the fruit without hitting current walls, BFS will be used 
def bfs(start,end,map,width,height):
    dict = {}
    act = {}
    for i in range(width):
        for j in range(height):
            if map.cell_state((i,j)) == SnakeCellState.WALL:
                dict[(i,j)] = 1
            else:
                dict[(i,j)] = 0
    solution= False
    Q = queue.Queue()
    Q.put(start)
    dict[start] = 1
    t = 0
    while not Q.empty():
        square = Q.get()

        for j,i in enumerate([(square[0]-1,square[1]),(square[0]+1,square[1]),(square[0],square[1]+1),(square[0],square[1]-1)]):
            if i[0] < 0 or i[0] >= width or i[1] < 0 or i[1] >= height:
                continue
            #LEFT,RIGHT,UP,DOWN
            if dict[i] == 1:
                continue
            act[i] = [j,square]
            if i == end:
                solution = True
                break
            Q.put(i)
            dict[i] = 1
        if solution:
            break
    if solution:
        point = end
        count = 0
        L= []
        while point != start:
            L.append(act[point][0])
            point = act[point][1]
            count += 1
        return [count,list(reversed(L))]
    #In case BFS doesn't work, commit suicide
    else: 
        return -1

#This is where the game will be played 
total = []
for i in range(10):
    env.reset()
    state = 0
    r = 0
    k = bfs(env.game.head(),env.game.dot,env.game,env.width,env.height)
    while True:
        env.render()
        #In case of no solution (snake is trapped), I make the snake suicide 
        if k == -1 or state >= len(k[1]):
            observation, reward, done, info = env.step(0)
        else:
            observation, reward, done, info = env.step(k[1][state])
        #time.sleep(0.1)
        
        if reward > 0:
            state = 0
            k = bfs(env.game.head(),env.game.dot,env.game,env.width,env.height)
        else:
            state += 1
        r += reward
        #termination
        if done:
            print('episode {} finished with {} reward'.format(i,r))
            total.append(r)
            break

plt.plot(range(1,11),total)
plt.show()


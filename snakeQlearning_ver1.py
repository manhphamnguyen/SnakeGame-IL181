import gym
import gym_snake
import random
import time
import queue
import copy

from gym_snake.envs.snake_env import SnakeCellState
env = gym.make('snake-v0')
save = 0

#Move state function
#Make the simulated move (not an official move in the last for loop)
def T(s,a):
    t = s
    t.game.step(a)
    return t

#If I simulated without saving like this, no save will be available, so this is necessary
#Tell me if you have a better way
def create_from_scratch(s):
    t = gym.make('snake-v0')
    t.game.snake = copy.deepcopy(s.game.snake)
    t.game.dot = copy.deepcopy(s.game.dot)
    t.game.empty_cells = copy.deepcopy(s.game.empty_cells)
    t.game.prev_action = copy.deepcopy(s.game.prev_action)
    return t

#check if snake will surely die 
def wall(game):
    (i,j) = game.head()
    t = [(i-1,j),(i+1,j),(i,j+1),(i,j-1)]
    for l in t:
        if game.cell_state(l) != 1 or l == game.snake[-1]:
            return False
    return True

#reward function
def R(s,a):
    (i,j) = s.game.head()
    t = [(i-1,j),(i+1,j),(i,j+1),(i,j-1)]
    l = t[a]
    dot = s.game.dot
    x = 0.3*(80-abs(l[0]-dot[0])-abs(l[1]-dot[1]))
    #death means -8000
    if l[0] < 0 or l[1] < 0 or l[0] > 39 or l[1] > 39:
        return -8000
    if s.game.cell_state(l) == 1:
        if l != s.game.snake[-1] or l == s.game.snake[1]:
            return -8000
    #eat the fruit, much higher than normal move
    elif s.game.cell_state(l) == 2:
        return 80+len(s.game.snake)
    #the closer to the fruit, more reward, there could be a better way, not sure
    return -0.1+x

def Q_learning(s):
    global save
    Q = {}
    k = gym.make('snake-v0')
    f = copy.deepcopy(s)
    alpha = 0.2
    gamma = 0.3
    l = []
    r = 0
    count,count2 = 0,0
    #time.sleep(0.5)
    while r < 80:
        #s.render() #you can render, but it willl be messy
        
        #initialize Q values
        if s.game.map() not in Q:
            Q[s.game.map()] = [R(s,i)+random.gauss(0,1) for i in range(4)]
        map = s.game.map()

        #These are standard QL algorithm
        if random.randint(1,10) == 1:
            a = random.randint(0,3)
        else:
            a = Q[s.game.map()].index(max(Q[s.game.map()]))
        r = R(s,a)
        t = T(s,a)
        if t.game.map() not in Q:
            Q[t.game.map()] = []
            for i in range(4):
                Q[t.game.map()].append(R(t,i)+random.gauss(0,1))
        Q[s.game.map()][a] = (1-alpha)*Q[s.game.map()][a]+alpha*(r+gamma*max(Q[t.game.map()]))
        s = t
        l.append(a)
        #no solutions after too many trials, restart completely  
        if count > 200:
            break
        #failed attempt or took too long, try again 
        if r == -8000 or count2 > 800:
            #('fail',count)
            count += 1
            s = f
            count2 = 0
            f = copy.deepcopy(s)
            l = []
        count2 += 1
    save += 1
    return l #return simulated move that is correct if there is one

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
        #time.sleep(1)
    if solution:
        point = end
        count = 0
        L= []
        while point != start:
            L.append(act[point][0])
            point = act[point][1]
            count += 1
        return [count,list(reversed(L))]
    #In case BFS doesn't work, Q-Learning will then be use 
    else: 
        mock = create_from_scratch(env)
        return [-1,Q_learning(mock)]

#This is where the game will be played 
for i in range(100):
    env.reset()
    state = 0
    r = 0
    k = bfs(env.game.head(),env.game.dot,env.game,env.width,env.height)
    while True:
        env.render()
        #In case of no solution, the return will sometimes crazy
        #and may crash the game, so in this case I make the snake suicide 
        if state >= len(k[1]):
            observation, reward, done, info = env.step(0)
        else:
            observation, reward, done, info = env.step(k[1][state])
        
        #The snake has eaten the fruit, move on to next fruit, total restart
        #You can make Q global instead, but it won't change much since there are too many possible cases
        if reward == 5:
            state = 0
            k = bfs(env.game.head(),env.game.dot,env.game,env.width,env.height)
        else:
            state += 1
        r += reward
        #termination
        if done:
            print('episode {} finished with {} reward'.format(i,r))
            break
#How many times has Q-Learning be useful out of all the episodes
print(save)


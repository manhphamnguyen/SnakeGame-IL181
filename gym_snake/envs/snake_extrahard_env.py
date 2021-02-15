from collections import Counter
from collections import deque
import random
import time
import gym
from gym import error, spaces, utils
from gym.envs.classic_control import rendering
from gym.utils import seeding

class SnakeAction(object):
    LEFT = 0
    RIGHT = 2
    UP = 1

class SnakeCellState(object):
    EMPTY = 0
    WALL = 1
    DOT = 2
    HEAD = 3

class SnakeReward(object):
    ALIVE = -0.1
    DOT = 5
    DEAD = -100
    WON = 100

class SnakeGame(object):
    def __init__(self, width, height, head):
        self.width = width
        self.height = height

        self.snake = deque()
        self.empty_cells = {(x, y) for x in range(width) for y in range(height)}
        self.dot = None

        self.prev_action = SnakeAction.UP
        self.prev_head = (head[0],head[1]-1)
        self.add_to_head(head)
        self.generate_dot()

    def add_to_head(self, cell):
        self.snake.appendleft(cell)
        if cell in self.empty_cells:
            self.empty_cells.remove(cell)
        if self.dot == cell:
            self.dot = None

    def cell_state(self, cell):
        if cell in self.empty_cells:
            return SnakeCellState.EMPTY
        if cell == self.dot:
            return SnakeCellState.DOT
        if cell == self.snake[0]:
            return SnakeCellState.HEAD
        return SnakeCellState.WALL
    
    def map(self):
        map = [tuple([self.cell_state((i,j)) for i in range(self.width)]) for j in range(self.height)]
        map = tuple(map)
        return map
        
    def head(self):
        return self.snake[0]

    def remove_tail(self):
        tail = self.snake.pop()
        self.empty_cells.add(tail)

    def can_generate_dot(self):
        return len(self.empty_cells) > 0

    def generate_dot(self):
        self.dot = random.sample(self.empty_cells, 1)[0]
        self.empty_cells.remove(self.dot)

    def next_head(self, action):
        head_x, head_y = self.head()
        t = [(head_x - 1, head_y),(head_x, head_y + 1),(head_x + 1, head_y),(head_x, head_y - 1)]
        next = self.prev_head
        for i in range(4):
            if next == t[i]:
                list = [t[(i+1)%4],t[(i+2)%4],t[(i+3)%4]]
                break
        if action == SnakeAction.LEFT:
            return list[0]
        if action == SnakeAction.RIGHT:
            return list[2]
        return list[1]

    def step(self, action):
        self.prev_action = action
        next_head = self.next_head(action)
        next_head_state = self.cell_state(next_head)
        self.prev_head = self.head()

        if next_head_state == SnakeCellState.WALL and next_head != self.snake[-1]:
            return SnakeReward.DEAD

        self.add_to_head(next_head)
            
        if next_head_state == SnakeCellState.DOT:
            if self.can_generate_dot():
                self.generate_dot()
                return SnakeReward.DOT                
            return SnakeReward.WON
            
        self.remove_tail()
            
        return SnakeReward.ALIVE

class SnakeExtraHardEnv(gym.Env):
    metadata= {'render.modes': ['human']}

    # TODO: define observation_space
    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.observation_space = None

        self.width = 40
        self.height = 40
        self.start = (10, 10)

        self.game = SnakeGame(self.width, self.height, self.start)
        self.viewer = None

    # TODO: define observation, info
    def step(self, action):
        reward = self.game.step(action)

        done = reward in [SnakeReward.DEAD, SnakeReward.WON]
        if reward == SnakeReward.WON:
            observation = True
        else:
            observation = False
        info = None
        #time.sleep(0.1)
        return observation, reward, done, info

    # TODO: define observation
    def reset(self):
        self.game = SnakeGame(self.width, self.height, self.start)
        observation = None
        return observation

    def render(self, mode='human', close=False):
        width = height = 600
        width_scaling_factor = width / self.width
        height_scaling_factor = height / self.height

        if self.viewer is None:
            self.viewer = rendering.Viewer(width, height)

        for x, y in self.game.snake:
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(0, 0, 0)
            self.viewer.add_onetime(square)

        if self.game.dot:
            x, y = self.game.dot
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(1, 0, 0)
            self.viewer.add_onetime(square)
            
        if self.game.head():
            x, y = self.game.head()
            l, r, t, b = x*width_scaling_factor, (x+1)*width_scaling_factor, y*height_scaling_factor, (y+1)*height_scaling_factor
            square = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            square.set_color(0, 1, 0)
            self.viewer.add_onetime(square)
            
        return self.viewer.render(return_rgb_array=mode=='rgb_array')

    def close(self):
        pass

    def seed(self):
        pass


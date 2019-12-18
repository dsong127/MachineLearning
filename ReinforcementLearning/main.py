import numpy as np
from random import randint
from copy import deepcopy
from matplotlib import pyplot as plt
#import math

discount_rate = 0.9
learning_rate = 0.2
grid_rows = 10
grid_cols = 10
#nb_cans = math.floor((grid_rows*grid_cols) / 2)
nb_cans = 50
nb_episodes = 5000
nb_moves = 200


epsilon = 1
epsilon_test = 0.1

Q_table = np.zeros((243, 5))

class World():
    def __init__(self, rows, cols):
        self.grid = self.initialize_grid(rows, cols)

    def initialize_grid(self, rows, cols):
        grid = np.zeros(rows*cols, dtype=int)
        r_indices = np.random.choice((rows*cols), nb_cans, replace=False)
        grid[r_indices] = 1
        grid = grid.reshape(rows, cols)

        assert(grid.shape == (grid_rows, grid_rows))
        assert(np.count_nonzero(grid) == 50)
        return grid

    def reset_grid(self, rows, cols):
        grid = np.zeros(rows*cols, dtype=int)
        r_indices = np.random.choice((rows*cols), nb_cans, replace=False)
        grid[r_indices] = 1
        grid = grid.reshape(rows, cols)
        assert(grid.shape == (grid_rows, grid_rows))
        assert(np.count_nonzero(grid) == 50)
        self.grid = grid

class Robot():
    def __init__(self):
        self.location = {'row': randint(0,9), 'col': randint(0,9)}
        self.current_state = [0,0,0,0,0]
        self.points = 0
        self.current_nb_moves = 0

    def reset_location(self):
        # Resets to random location
        rr = randint(0,9)
        rc = randint(0,9)
        self.location['row'] = rr
        self.location['col'] = rc

    def choose_perform_action(self,grid, test=False):
        prev_state = deepcopy(self.current_state)
        q_row = ter_to_dec(self.current_state)

        e = epsilon if test == False else epsilon_test
        # Action: 0: pick up, 1: left, 2: right, 3: up, 4: down
        if np.random.uniform(0, 1) <= e:
            action = randint(0, 4) # Some action
        else:
            action = np.argmax(Q_table[q_row, :])

        reward = 0

        # Pick up
        # If there is a can, get points and change 1 to 0
        if action == 0:
            if grid[self.location['row'], self.location['col']] == 1:
                reward = 10
                grid[self.location['row'], self.location['col']] = 0
            else:
                reward = -1
        # If there is a wall, -5 points, and stay in same spot
        # else move to a new location
        elif action == 1:
            if self.location['col'] == 0:
                reward = -5
            else:
                self.location['col'] -= 1

        elif action == 2:
            if self.location['col'] == 9:
                reward = -5
            else:
                self.location['col'] += 1

        elif action == 3:
            if self.location['row'] == 0:
                reward = -5
            else:
                self.location['row'] -= 1

        elif action == 4:
            if self.location['row'] == 9:
                reward = -5
            else:
                self.location['row'] += 1

        self.current_nb_moves += 1
        self.points += reward
       
        return prev_state, reward, action

    def update_q(self, p_state, p_action, reward):
        p_row = ter_to_dec(p_state)
        current_row = ter_to_dec(self.current_state)
        predict_best = np.max(Q_table[current_row, :])
        p_q = Q_table[p_row, p_action]
        
        Q_table[p_row, p_action] += learning_rate * (float(reward) + discount_rate * predict_best - float(p_q))
        return


    def perceive_current_state(self, grid):
        left = 2 if self.location['col'] == 0 else grid[self.location['row']][self.location['col']-1]
        right = 2 if self.location['col'] == 9 else grid[self.location['row']][self.location['col']+1]
        down = 2 if self.location['row'] == 9 else grid[self.location['row']+1][self.location['col']]
        up = 2 if self.location['row'] == 0 else grid[self.location['row']-1][self.location['col']]
        here = grid[self.location['row'], self.location['col']]

        state_dict = {'here': here, 'left': left, 'right': right, 'up': up, 'down': down}
        current_state = dict_to_ter(state_dict)

        self.current_state = current_state

def dict_to_ter(state):
    ter_arr = []
    for v in state.values():
        ter_arr.append(v)
    return np.array(ter_arr)

def ter_to_dec(ter_arr):
    arr = deepcopy(ter_arr)
    for i in range(4):
        arr[i] = ter_arr[i] * (3 ** (4 -i))
    q_value = np.sum(arr)
    return q_value

def train():
    global epsilon
    print('Starting state:')
    world = World(grid_rows, grid_cols)
    print(world.grid)
    rob = Robot()
    print(rob.location)
    rob.perceive_current_state(world.grid)
    print('current state: {}'.format(rob.current_state))
    total_points = []

    for i in range(nb_episodes):
        print('i: {} \t state: {}'.format(i, rob.current_state))
        while rob.current_nb_moves <= 200:
            prev_state, reward, action = rob.choose_perform_action(world.grid)
            rob.perceive_current_state(world.grid)
            rob.update_q(prev_state, action, reward)
        # Decrease epsilon every 50 epoch
        if (i + 1) % 50 == 0 and epsilon > 0.1:
            epsilon -= 0.01
        # Reset points at the end of episode
        if (i+1) % 100 == 0:
            total_points.append(rob.points)
        # Reset errthang
        rob.points = 0
        rob.current_nb_moves = 0
        rob.reset_location()
        world.reset_grid(grid_rows, grid_cols)

    plt.figure(figsize=(10, 10))
    x = range(0, 5000, 100)
    plt.title('Training reward')
    plt.plot(x, total_points)
    plt.xlabel("Episodes")
    plt.ylabel("Total Points")
    plt.savefig('plot.png', bbox_inches='tight')

def test():
    world = World(grid_rows, grid_cols)
    rob = Robot()
    rob.perceive_current_state(world.grid)
    total_points = []

    for i in range(nb_episodes):
        print('i: {} \t state: {}'.format(i, rob.current_state))
        while rob.current_nb_moves <= 200:
            prev_state, reward, action = rob.choose_perform_action(world.grid, test=True)
            rob.perceive_current_state(world.grid)
            rob.update_q(prev_state, action, reward)
        if (i + 1) % 100 == 0:
            total_points.append(rob.points)
        # Reset errthang
        rob.points = 0
        rob.current_nb_moves = 0
        rob.reset_location()
        world.reset_grid(grid_rows, grid_cols)

    average = np.average(total_points)
    std = np.std(total_points)

    plt.figure(figsize=(10, 10))
    x = range(0, 5000, 100)
    plt.title('Testing reward')
    plt.plot(x, total_points)
    plt.xlabel("Episodes")
    plt.ylabel("Total Points")
    plt.savefig('plot_test.png', bbox_inches='tight')

    print(average)
    print(std)

if __name__ == '__main__':
    train()
    test()

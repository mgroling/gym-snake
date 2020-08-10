import gym
import numpy as np
import math

class SnakeEnv(gym.Env):
    """
    Description:
        A snake manuevers through a world with the goal to find and eat apples, in order to get bigger.
        While doing so it avoids any obstacles in its path including itself, to prevent a horrible death.

    Source:
        This environment corresponds to the popular video game concept.

    Observation:
        There are three modes for observation: "image", "ray", "simple":

        Image:
            Type: Box(size, size)
            This returns a 2d numpy array with pixel format.
            0 is an empty pixel.
            1 is a pixel with a part of the snake on it.
            2 is a pixel with an apple on it.

        Ray:
            Type: Box(8, 3)
            This returns raycasts, with center position being the head of the snake.
            First column are raycasts of the snake body, second column are raycasts of apples and third column is raycasts of walls.
            Rows corresponds to the direction, snake can see in 8 directions (cardinal points: N, NE, E, SE, S, SW, W, NW)
            If the corresponding type of object is very close, the value for the raycast is close to 1 and if it is far away it is close to zero.

        Simple:
            Type: Box(10)
            This returns if the next block is an obstacle. This is done in all 8 directions (like in Ray).
            0 is not an obstacle
            1 is an obstacle
            The last 2 variables display the vector from the snake head to the apple, scaled down between zero and one. One being close, zero being far.

    Actions:
        Type: Discrete(4)
        Num Action
        0   Snake goes up
        1   Snake goes right
        2   Snake goes down
        3   Snake goes left

    Reward:
        Reward can be set by giving an iterable object.
        Event                                               Reward
        Snake gets closer to apple, but does not eat it     reward[1], default = 0
        Snake eats apple                                    reward[2], default = 1
        Snake dies                                          reward[3], default = -1
        Anything else                                       reward[0], default = 0

    Starting state:
        Starting state can be eihter set to "center" or "random".

    Episode Termination:
        Snake dies by sliding into itself or a wall.
        Snake has not eaten an apple since 25 timesteps. This can be customised with termination.

    Further Customisation:
        "size" changes the amount of blocks in height and width.
        "add_len" changes the additional length the snake gets by eating an apple.
        "start_length" changes the starting length.
    """
    def __init__(self, size = 15, reward = (0, 0, 1, -1), obs = "image", spawn = "center", add_len = 1, termination = 50, start_length = 3, seed = None):
        assert type(size) == int and size > 0, "Invalid parameter: size must be an integer and greater than zero"
        try:
            assert len(reward) >= 4, "Invalid parameter: reward must be iterable and contain four or more elements, (all numbers)"
        except:
            assert False, "Invalid parameter: reward must be iterable and contain four or more elements, (all numbers)"
        assert obs in ["image", "ray", "simple"], "Invalid parameter: obs must be either \"image\", \"ray\" or \"simple\""
        assert spawn in ["center", "random"], "Invalid parameter: spawn must be either \"center\" or \"random\""
        assert type(add_len) == int and add_len >= 0, "Invalid parameter: add_len must be an integer and greater-or-equal than zero"
        assert type(start_length) == int and start_length > 0, "Invalid parameter: start_length must be an integer and gerater than zero"
        assert type(termination) == int and termination > 0, "Invalid parameter: termination must be an integer and greater than zero"

        self._size = size
        self._reward = reward
        self._obs = obs
        self._spawn = spawn
        self._add_len = add_len
        self._termination = termination
        self._start_len = start_length
        np.random.seed = seed

        #setup variables
        self._matrix = None
        self._snake = None
        self._done = True
        self._mov_possibilites = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        self._snake_len = None
        self._viewer = None
        self._snake_hunger = 0
        self._act_mapping = {0: np.array([-1, 0]), 1: np.array([0, 1]), 2: np.array([1, 0]), 3: np.array([0, -1])}

        self.action_space = gym.spaces.Discrete(4)
        if obs == "image":
            self.observation_space = gym.spaces.Box(np.zeros((self._size, self._size)), np.full((self._size, self._size), 2))
        elif obs == "ray":
            self.observation_space = gym.spaces.Box(np.zeros((8, 3)), np.ones((8, 3)))
        elif obs == "simple":
            self.observation_space = gym.spaces.Box(np.zeros(10), np.ones(10))

    def set_params(self, size = 15, reward = (0, 0, 1, -1), obs = "image", spawn = "center", add_len = 1, termination = 50, start_length = 3, seed = None):
        assert type(size) == int and size > 0, "Invalid parameter: size must be an integer and greater than zero"
        try:
            assert len(reward) >= 4, "Invalid parameter: reward must be iterable and contain four or more elements, (all numbers)"
        except:
            assert False, "Invalid parameter: reward must be iterable and contain four or more elements, (all numbers)"
        assert obs in ["image", "ray", "simple"], "Invalid parameter: obs must be either \"image\", \"ray\" or \"simple\""
        assert spawn in ["center", "random"], "Invalid parameter: spawn must be either \"center\" or \"random\""
        assert type(add_len) == int and add_len >= 0, "Invalid parameter: add_len must be an integer and greater-or-equal than zero"
        assert type(start_length) == int and start_length > 0, "Invalid parameter: start_length must be an integer and gerater than zero"
        assert type(termination) == int and termination > 0, "Invalid parameter: termination must be an integer and greater than zero"

        self._size = size
        self._reward = reward
        self._obs = obs
        self._spawn = spawn
        self._add_len = add_len
        self._termination = termination
        self._start_len = start_length
        np.random.seed = seed

        if obs == "image":
            self.observation_space = gym.spaces.Box(np.zeros((self._size, self._size)), np.full((self._size, self._size), 2))
        elif obs == "ray":
            self.observation_space = gym.spaces.Box(np.zeros((8, 3)), np.ones((8, 3)))
        elif obs == "simple":
            self.observation_space = gym.spaces.Box(np.full((10), -1), np.full((10), 2))

    def reset(self):
        #setup matrix
        self._matrix = np.zeros((self._size, self._size), dtype = np.int8)
        #spawn snake
        if self._spawn == "center":
            self._snake = []
            for i in range(0, self._start_len):
                self._snake.append((self._size//2, self._size//2-self._start_len//2+i))
        elif self._spawn == "random":
            self._snake = [((np.random.randint(self._size), np.random.randint(self._size)))]
            for i in range(1, self._start_len):
                direction = self._act_mapping[np.random.randint(4)]
                next_part = np.array([self._snake[-1][0]+direction[0], self._snake[-1][1]+direction[1]])
                while (next_part < 0).any() or (next_part >= self._size).any():
                    direction = self._act_mapping[np.random.randint(4)]
                    next_part = np.array([self._snake[-1][0]+direction[0], self._snake[-1][1]+direction[1]])
                self._snake.append((self._snake[-1][0]+direction[0], self._snake[-1][1]+direction[1]))
        for i in range(0, len(self._snake)):
            self._matrix[self._snake[i][0], self._snake[i][1]] = 1
        self._snake_len = self._start_len
        self._snake_hunger = 0

        self._done = False
        self._spawnApple()

        if self._obs == "image":
            return self._matrix
        elif self._obs == "ray":
            return self._getRays()
        elif self._obs == "simple":
            return self._getSimple()
        
    def step(self, action):
        assert action in [0, 1, 2, 3], "invalid action -- valid actions are: 0, 1, 2, 3"

        if self._done:
            gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

        action = self._act_mapping[action]

        reward = self._reward[0]
        new_head_pos = self._snake[0] + action

        old_apple_dist = self.getDistance(self._snake[0], self._apple)
        new_apple_dist = self.getDistance(new_head_pos, self._apple)
        reward = self._reward[1] if old_apple_dist > new_apple_dist else self._reward[0]

        #moved outside
        if (new_head_pos < 0).any() or (new_head_pos >= self._size).any():
            self._done = True
            reward = self._reward[3]
        #moved into apple
        elif self._matrix[new_head_pos[0], new_head_pos[1]] == 2:
            self._snake_len += self._add_len
            self._spawnApple()
            reward = self._reward[2]
            self._snake_hunger = 0
        #moved into itself
        elif self._matrix[new_head_pos[0], new_head_pos[1]] == 1:
            self._done = True
            reward = self._reward[3]
        
        if not self._done:
            #move head
            self._matrix[new_head_pos[0], new_head_pos[1]] = 1

        #remove tail if snake has not gotten any bigger
        if len(self._snake) >= self._snake_len:
            self._matrix[self._snake[-1][0], self._snake[-1][1]] = 0
            self._snake.pop()

        self._snake.insert(0, tuple(new_head_pos))

        #snake has not eaten something for too long
        if self._snake_hunger > self._termination:
            self._done = True
            reward = self._reward[3]

        self._snake_hunger += 1
        
        if self._obs == "image":
            return self._matrix, reward, self._done, {}
        elif self._obs == "ray":
            return self._getRays(), reward, self._done, {}
        elif self._obs == "simple":
            return self._getSimple(), reward, self._done, {}

    def render(self, mode = "human"):
        screen_size = 800 - self._size*1

        if self._viewer is None:
            from gym.envs.classic_control import rendering
            self._viewer = rendering.Viewer(screen_size, screen_size)

            #create squares to represent the game
            length_square = (screen_size+self._size*1) / self._size - 1
            self._squares = [[rendering.FilledPolygon([(i*length_square+1, j*length_square+1), ((i+1)*length_square-1, j*length_square+1), 
                                        ((i+1)*length_square-1, (j+1)*length_square-1), (i*length_square+1, (j+1)*length_square-1)])
                                        for i in range(0, self._size)] for j in range(0, self._size)]
            for i in range(0, self._size):
                for j in range(0, self._size):
                    self._viewer.add_geom(self._squares[i][j])

        if self._matrix is None:
            return None
        
        #color squares accordingly
        temp = np.flip(self._matrix, 0)
        for i in range(0, self._size):
            for j in range(0, self._size):
                if temp[i, j] == 0:
                    self._squares[i][j].set_color(0, 0, 0)
                elif temp[i, j] == 1:
                    self._squares[i][j].set_color(0, 255, 255)
                elif temp[i, j] == 2:
                    self._squares[i][j].set_color(255, 0, 0)        

        return self._viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None

    def _spawnApple(self):
        #search for an empty block randomly and insert apple there
        rand_x, rand_y = np.random.randint(self._size), np.random.randint(self._size)
        while self._matrix[rand_x, rand_y] != 0:
            rand_x, rand_y = np.random.randint(self._size), np.random.randint(self._size)
        self._matrix[rand_x, rand_y] = 2
        self._apple = (rand_x, rand_y)

    def _getRays(self):
        head = self._snake[0][0], self._snake[0][1]
        rays = np.zeros((8, 3))
        index = 0
        #loop through all directions
        for i, j in self._mov_possibilites:
            k = 1
            snakeFound, appleFound = False, False
            pos = np.array([head[0] + i*k, head[1] + j*k])
            #go one step at a time into a direction and check the value of matrix there
            #went outside of the playing field
            while (pos >= 0).all() and (pos < self._size).all():
                if self._matrix[pos[0], pos[1]] == 1 and not snakeFound:
                    rays[index, 0] = 1-(k-1)/(self._size-1)
                    snakeFound = True
                elif self._matrix[pos[0], pos[1]] == 2 and not appleFound:
                    rays[index, 1] = 1-(k-1)/(self._size-1)
                    appleFound = True

                k += 1
                pos = np.array([head[0] + i*k, head[1] + j*k])

            rays[index, 2] = 1 - (k-1)/(self._size-1)
            index += 1
        
        return rays

    def _getSimple(self):
        dist_apple_x_y = np.array([1 - (self._apple[0]-self._snake[0][0])/(self._size-1), 1 - (self._apple[1]-self._snake[0][1])/(self._size-1)])

        head = self._snake[0][0], self._snake[0][1]
        next_view = np.zeros((8,1))
        index = 0
        for i, j in self._mov_possibilites:
            if head[0]+i >= 0 and head[1]+j >= 0 and head[0]+i < self._size and head[1]+j < self._size:
                if self._matrix[head[0]+i, head[1]+j] == 0 or self._matrix[head[0]+i, head[1]+j] == 2:
                    next_view[index] = 0
                else:
                    next_view[index] = 1
            else:
                next_view[index] = 1
            index += 1

        return np.append(next_view, dist_apple_x_y)

    def getDistance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
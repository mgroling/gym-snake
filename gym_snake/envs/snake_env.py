import gym
import numpy as np

class SnakeEnv(gym.Env):
    def __init__(self, width = 25, height = 25, reward = "hard", obs = "image", spawn = "center", add_len = 1, num_apples = 1, seed = None):
        self._matrix = None
        self._snake = None
        self._snake_len = None
        self._done = True

        self._width = width
        self._height = height
        self._reward = reward
        self._obs = obs
        self._spawn = spawn
        self._add_len = add_len
        self._num_apples = num_apples
        np.random.seed = seed

        self.action_space = gym.spaces.Discrete(4)
        # if obs == "image":
        #     self.observation_space = spaces.Box()
        # elif obs == "simple":
        #     self.observation_space = gym.spaces.Box()

    def reset(self):
        self._matrix = np.zeros((self._width, self._height), dtype = np.int8)
        if self._spawn == "center":
            self._snake = [(self._width//2, self._height//2)]
        elif self._spawn == "random":
            self._snake = [((np.random.randint(self._width), np.random.randint(self._height)))]
        self._matrix[self._snake[0][0], self._snake[0][1]] = 1
        self._snake_len = 1
        self._done = False
        #create apples
        for i in range(0, self._num_apples):
            self._spawnApple()

        if self._obs == "image":
            return self._matrix
        elif self._obs == "ray":
            return self._getRays()
        elif self._obs == "simple":
            return self._getSimple()
        
    def step(self, action):
        assert action in [0, 1, 2, 3], "invalid action"

        if self._done:
            gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )

        if action == 0:
            action = np.array([0, 1])
        elif action == 1:
            action = np.array([1, 0])
        elif action == 2:
            action = np.array([0, -1])
        elif action == 3:
            action = np.array([-1, 0])

        new_head_pos = self._snake[0] + action
        #moved outside
        if (new_head_pos < 0).any() or new_head_pos[0] >= self._width or new_head_pos[1] >= self._height:
            self._done = True
        #moved into apple
        elif self._matrix[new_head_pos[0], new_head_pos[1]] == 2:
            self._snake_len += self._add_len
            self._spawnApple()
        #moved into itself
        elif self._matrix[new_head_pos[0], new_head_pos[1]] == 1:
            self._done = True
        
        if not self._done:
            #move head
            self._matrix[new_head_pos[0], new_head_pos[1]] = 1

        #snake has not gotten any bigger
        if len(self._snake) >= self._snake_len:
            self._matrix[self._snake[-1][0], self._snake[-1][1]] = 0
            self._snake.pop()

        if not self._done:
            self._snake.insert(0, tuple(new_head_pos))

        if self._obs == "image":
            return self._matrix

    def _spawnApple(self):
        rand_x, rand_y = np.random.randint(self._width), np.random.randint(self._height)
        while self._matrix[rand_x, rand_y] != 0:
            rand_x, rand_y = np.random.randint(self._width), np.random.randint(self._height)
        self._matrix[rand_x, rand_y] = 2

# env = SnakeEnv(width = 3, height = 3)
# env.reset()
# print(env._matrix)
# print(env.step(2))
# print(env.step(2))
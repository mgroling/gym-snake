from gym.envs.registration import register

register(
    id='gym-snake-v0',
    entry_point='gym_snake.envs:SnakeEnv',
)
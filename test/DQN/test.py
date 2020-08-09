import gym
import gym_snake
import time

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                           layers=[128, 64, 32],
                                           layer_norm=True,
                                           feature_extraction="mlp")

def main():
    env = gym.make("gym-snake-v0")
    env.set_params(reward = "hard", obs = "ray", size = 15, termination = 75, spawn = "random", add_len = 3)

    # model = DQN.load("test/DQN/deepq_snake")
    # model.set_env(env)

    # model = DQN(CustomDQNPolicy, env, verbose=1)

    # model.learn(total_timesteps = 100000)

    # model.save("test/DQN/deepq_snake_ray_3grow")

    model = DQN.load("test/DQN/deepq_snake_ray_3grow")

    obs = env.reset()

    """testing"""
    reward_sum = 0.0
    obs = env.reset()
    for i in range(0, 10):
        timestep = 0
        done = False
        while not done:
            timestep += 1
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            time.sleep(0.05)
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
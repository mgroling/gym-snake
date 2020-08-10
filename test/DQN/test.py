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
                                           layers=[256, 64],
                                           layer_norm=True,
                                           feature_extraction="mlp")

def main():
    env = gym.make("gym-snake-v0")
    env.set_params(reward = (0, 0, 1, -1), obs = "ray", size = 15, termination = 150, spawn = "random", add_len = 1)

    # model = DQN.load("test/DQN/deepq_snake")
    # model.set_env(env)

    # model = DQN(CustomDQNPolicy, env, verbose=1)

    # model.learn(total_timesteps = 100000)

    # model.save("test/DQN/deepq_snake_ray_5grow_100k_10death")

    model = DQN.load("test/DQN/deepq_snake_ray_5grow_300k")

    obs = env.reset()

    """testing"""
    reward_sum = 0.0
    obs = env.reset()
    for i in range(0, 10):
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            time.sleep(0.01)
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

    env.close()

if __name__ == "__main__":
    main()
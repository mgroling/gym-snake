import gym
import gym_snake
import time

from stable_baselines3 import DQN

if __name__ == "__main__":
    env = gym.make("gym-snake-v0")
    env.set_params(
        reward=(0, 0, 1, -1),
        obs="ray",
        size=10,
        termination=100,
        spawn="random",
        add_len=1,
    )

    train = True
    model_path = "experiments/DQN/models/model_ray"

    if train:
        model = DQN("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=500000)
        model.save(model_path)
    else:
        model = DQN.load(model_path)
        model.set_env(env)
        obs = env.reset()

        """testing"""
        reward_sum = 0.0
        obs = env.reset()
        for i in range(0, 10):
            done = False
            while not done:
                action, _ = model.predict(obs)
                action = float(action)
                obs, reward, done, _ = env.step(action)
                reward_sum += reward
                env.render()
                time.sleep(0.1)
            print(reward_sum)
            reward_sum = 0.0
            obs = env.reset()

        env.close()

import gym
import gym_snake
import time
import torch
import torch.nn as nn
import numpy as np

from abc_alg import ABC


def predict(model, x):
    return int(torch.argmax(model(torch.from_numpy(x))))


def setWeights(model, weights):
    offset = 0
    with torch.no_grad():
        for param in model.parameters():
            shape = param.data.shape
            num_weights = shape[0] * shape[1]
            param.data = nn.parameter.Parameter(
                torch.from_numpy(
                    weights[offset : offset + num_weights].reshape((shape[0], shape[1]))
                )
            )
            offset += num_weights


def getNumWeights(model):
    num_weights = 0
    with torch.no_grad():
        for param in model.parameters():
            print(param.data.shape)
            shape = param.data.shape
            num_weights += shape[0] * shape[1]
    return num_weights


def objective_single(env, model, x, render=False):
    setWeights(model, x)
    obs = env.reset()
    total_reward = 0

    done = False
    while not done:
        action = predict(model, obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        if render:
            env.render()
            time.sleep(0.1)

    return -total_reward


def objective(env, model, weights):
    rewards = np.zeros(
        (
            len(
                weights,
            )
        )
    )
    for i in range(len(weights)):
        rewards[i] = objective_single(env, model, weights[i], render=False)

    return rewards


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Discrete(24)

    def observation(self, state):
        return state.reshape((24,))


if __name__ == "__main__":
    # model 1:
    # COLONY_SIZE = 300
    # NUM_CYCLES = 1000
    # SCOUT_LIMIT_START = 50
    # SCOUT_LIMIT_END = 300
    # model 2:
    COLONY_SIZE = 300
    NUM_CYCLES = 3000
    SCOUT_LIMIT_START = 500
    SCOUT_LIMIT_END = 500

    MODEL_STRUC = 0

    if MODEL_STRUC == 0:
        model = nn.Sequential(nn.Linear(10, 4, bias=False))
        save_path = "experiments/ABC/weights_1layer.npy"
    elif MODEL_STRUC == 1:
        model = nn.Sequential(
            nn.Linear(10, 24, bias=False), nn.ReLU(), nn.Linear(24, 4, bias=False)
        )
        save_path = "experiments/ABC/weights_2layer.npy"

    train = True
    # save_path = "experiments/ABC/weights_1layer_model1.npy"
    env = gym.make("gym-snake-v0")
    # env = ObservationWrapper(env)
    env.set_params(
        reward=(0, 0.01, 1, -1),
        obs="simple",
        size=10,
        termination=75,
        spawn="random",
        add_len=1,
        start_length=3,
    )

    if train:
        num_weights = getNumWeights(model)
        abc = ABC(
            num_cycles=NUM_CYCLES,
            observation_shape=(num_weights,),
            objective=lambda x: objective(env, model, x),
            colony_size=COLONY_SIZE,
            scout_limit_start=SCOUT_LIMIT_START,
            scout_limit_end=SCOUT_LIMIT_END,
        )
        weights = abc.run(save_path=save_path)

        for i in range(5):
            objective_single(env, model, weights, render=True)
    else:
        weights = np.load(save_path)
        for i in range(10):
            objective_single(env, model, weights, render=True)

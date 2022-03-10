from random import sample
import gym
import gym_snake
import time
import torch
import torch.nn as nn
import numpy as np
from ContinuousChromosome import ContinuousChromosome, ContinuousRepresentation
from GA import GA


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


def objective(env, model, x, render=False):
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

    return -total_reward  # minimize


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Discrete(24)

    def observation(self, state):
        return state.reshape((24,))


if __name__ == "__main__":
    # models:
    # weights: (Linear(10, 4, bias = False), ReLU), (pop: 300, gens: 200)
    # weights_2layer: (Linear(10, 24, bias = False), ReLU, Linear(24, 4, bias = False)), (pop: 300, gens: 400)
    train = False
    env = gym.make("gym-snake-v0")
    # env = ObservationWrapper(env)
    env.set_params(
        reward=(0, 0, 1, -1),
        obs="simple",
        size=15,
        termination=75,
        spawn="random",
        add_len=1,
        start_length=3,
    )
    model = nn.Sequential(
        nn.Linear(10, 24, bias=False), nn.ReLU(), nn.Linear(24, 4, bias=False)
    )

    if train:
        r = ContinuousRepresentation(-1, 1, getNumWeights(model))
        ga = GA(r, lambda x: objective(env, model, x), debug=1)
        ga.initPopulation(300)
        ga.evolve(400)
        best_weights = ga.getSolution()[1]
        for i in range(5):
            objective(env, model, best_weights, render=True)

        # save model
        np.save("experiments/GA/weights_2layer.npy", best_weights)
    else:
        weights = np.load("experiments/GA/weights_2layer.npy")
        for i in range(10):
            objective(env, model, weights, render=True)

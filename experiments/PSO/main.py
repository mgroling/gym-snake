import gym
import gym_snake
import time
import torch
import torch.nn as nn
import numpy as np

from pso import PSO


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

    return total_reward


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
    # PSO Parameters
    NUM_ITERATIONS = 1000
    POPULATION_SIZE = 100
    MIN_VELOCITY = -1
    MAX_VELOCITY = 1
    C_1 = 1
    C_2 = 1
    C_3 = 1
    RANDOM_FACTOR = 1
    W_START = 1
    W_END = 0.3

    MODEL_STRUC = 0

    if MODEL_STRUC == 0:
        model = nn.Sequential(nn.Linear(10, 4, bias=False))
        save_path = "experiments/PSO/weights_1layer.npy"
    elif MODEL_STRUC == 1:
        model = nn.Sequential(
            nn.Linear(10, 24, bias=False), nn.ReLU(), nn.Linear(24, 4, bias=False)
        )
        save_path = "experiments/PSO/weights_2layer.npy"

    train = True
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
        pso = PSO(
            [POPULATION_SIZE, num_weights],
            lambda x: objective(env, model, x),
            min_velocity=MIN_VELOCITY,
            max_velocity=MAX_VELOCITY,
            num_iterations=NUM_ITERATIONS,
            c_1=C_1,
            c_2=C_2,
            c_3=C_3,
            w_start=W_START,
            w_end=W_END,
            random_factor=RANDOM_FACTOR,
        )
        weights = pso.run(
            np.random.random((POPULATION_SIZE, num_weights)) - 0.5, save_path=save_path
        )

        for i in range(5):
            objective_single(env, model, weights, render=True)

        # save model
        np.save(save_path, weights)
    else:
        weights = np.load(save_path)
        for i in range(10):
            objective_single(env, model, weights, render=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import gym_snake
import time
import numpy as np


class ReplayBuffer:
    def __init__(self, max_size, env) -> None:
        if env.observation_space.shape == ():
            self.obs_t = np.empty((max_size, env.observation_space.n), dtype=np.float32)
            self.obs_tp1 = np.empty(
                (max_size, env.observation_space.n), dtype=np.float32
            )
        else:
            self.obs_t = np.empty(
                (max_size, *env.observation_space.shape), dtype=np.float32
            )
            self.obs_tp1 = np.empty(
                (max_size, *env.observation_space.shape), dtype=np.float32
            )

        self.action = np.empty((max_size,), dtype=np.float32)
        self.reward = np.empty((max_size,), dtype=np.float32)
        self.done = np.empty((max_size,), dtype=bool)
        self.max_size = max_size
        self.size = 0

    def add(self, obs_t, obs_tp1, action, reward, done):
        if self.size < self.max_size:
            self.obs_t[self.size] = obs_t
            self.obs_tp1[self.size] = obs_tp1
            self.action[self.size] = action
            self.reward[self.size] = reward
            self.done[self.size] = done
            self.size += 1
        else:
            # randomly replace a sample
            rand = np.random.randint(self.size)
            self.obs_t[rand] = obs_t
            self.obs_tp1[rand] = obs_tp1
            self.action[rand] = action
            self.reward[rand] = reward
            self.done[rand] = done

    def sample(self, num_samples):
        """randomly sample num_samples from the replay buffer"""
        assert num_samples <= self.size, "Not enough samples in the replay buffer"
        rand = np.random.choice(self.size, num_samples, replace=False)

        return (
            self.obs_t[rand],
            self.obs_tp1[rand],
            self.action[rand],
            self.reward[rand],
            self.done[rand],
        )


class Net(nn.Module):
    def __init__(self, gamma=0.99) -> None:
        super(Net, self).__init__()
        self.l1 = nn.Linear(10, 4)
        # self.l2 = nn.Linear(48, 4)
        self.gamma = gamma

    def forward(self, x):
        x = torch.from_numpy(x).float()
        x = self.l1(x)
        # x = self.l2(x)
        return x

    def predict(self, x):
        action = int(torch.argmin(self(x)))
        return action, None

    def train(
        self,
        env,
        epochs,
        batch_size=32,
    ):
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        buf = ReplayBuffer(10000, env)

        for epoch in range(epochs):
            for i in range(100):
                obs = env.reset()

                done = False
                while not done:
                    if np.random.rand() < 1 - (epoch / epochs) - 0.5:
                        action = np.random.randint(4)
                    else:
                        action, _ = model.predict(obs)

                    temp_obs = obs.copy()
                    obs, reward, done, _ = env.step(action)
                    buf.add(temp_obs, obs, action, reward, done)

            # train model if there are enough samples in the replay buffer
            if buf.size >= batch_size:
                for i in range(100):
                    obs_t, obs_tp1, action, reward, done = buf.sample(batch_size)

                    with torch.no_grad():
                        expected_rewards = torch.max(self(obs_tp1), 1)
                        total_reward = (
                            torch.from_numpy(reward)
                            + torch.from_numpy(np.logical_not(done))
                            * self.gamma
                            * expected_rewards.values
                        )

                    optimizer.zero_grad()
                    outputs = self(obs_t)
                    labels = outputs.clone()
                    for k in range(len(labels)):
                        labels[k, action] = total_reward[k]

                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            if epoch % 100 == 0:
                print("Epoch {} finished".format(epoch))


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super(ObservationWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Discrete(24)

    def observation(self, state):
        return state.reshape(24)


if __name__ == "__main__":
    env = gym.make("gym-snake-v0")
    # env = ObservationWrapper(env)
    env.set_params(
        reward=(0, 0, 1, -1),
        obs="simple",
        size=15,
        termination=150,
        spawn="random",
        add_len=1,
        start_length=3,
    )

    model = Net()
    model.float()

    model.train(env, 10000, batch_size=128)

    # save
    torch.save(model.state_dict(), "experiments/pytorch-DQN/model")

    reward_sum = 0.0
    obs = env.reset()
    for i in range(5):
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            env.render()
            time.sleep(0.1)
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

    env.close()

# Bat algorithm

import numpy as np

from typing import Callable, List


class BAT:
    def __init__(
        self,
        observation_shape: List[int],
        num_iterations: int,
        num_bats: int,
        objective: Callable[[np.ndarray], float],
        min_frequency: float = 0,
        max_frequency: float = 10,
        min_loudness: float = 0,
        max_loudness: float = 1,
        min_pulse_rate: float = 0,
        max_pulse_rate: float = 1,
        alpha: float = 0.1,
        gamma: float = 0.1,
    ) -> None:
        self.positions: np.ndarray = np.empty((num_bats, *observation_shape))
        self.fitness_values: np.ndarray = np.empty((num_bats,))
        self.velocities: np.ndarray = np.empty((num_bats, *observation_shape))
        self.frequencies: np.ndarray = np.empty((num_bats,))
        self.loudness: np.ndarray = np.empty((num_bats,))
        self.pulse_rate: np.ndarray = np.empty((num_bats,))
        self.best_position: np.ndarray = np.empty(observation_shape)
        self.best_value: float = -np.inf
        self.initial_pulse_rate: np.ndarray = np.empty((num_bats,))

        self.observation_shape: List[int] = observation_shape
        self.num_iterations: int = num_iterations
        self.num_bats: int = num_bats
        self.objective: Callable[[np.ndarray], float] = objective
        self.min_frequency: float = min_frequency
        self.max_frequency: float = max_frequency
        self.min_loudness: float = min_loudness
        self.max_loudness: float = max_loudness
        self.min_pulse_rate: float = min_pulse_rate
        self.max_pulse_rate: float = max_pulse_rate
        self.alpha: float = alpha
        self.gamma: float = gamma

    def update_best(self):
        best_index: int = np.argmax(self.fitness_values)
        if self.fitness_values[best_index] > self.best_value:
            self.best_position = self.positions[best_index]

    def run(self, save_path: str = None) -> None:
        self.positions = np.random.random((self.num_bats, *self.observation_shape))
        self.fitness_values = self.objective(self.positions)
        self.velocities = np.random.random((self.num_bats, *self.observation_shape))
        self.frequencies = self.min_frequency + (
            self.max_frequency - self.min_frequency
        ) * np.random.random((self.num_bats,))
        self.loudness = self.min_loudness + (
            self.max_loudness - self.min_loudness
        ) * np.random.random((self.num_bats,))
        self.pulse_rate = self.min_pulse_rate + (
            self.max_pulse_rate - self.min_pulse_rate
        ) * np.random.random((self.num_bats,))

        self.initial_pulse_rate = np.array(self.pulse_rate, copy=True)
        self.update_best()

        for i in range(self.num_iterations):
            self.velocities = (
                self.velocities
                + (self.positions - self.best_position)
                * self.frequencies[:, np.newaxis]
            )
            self.positions = self.positions + self.velocities

            # change bat positions to local position around global optimum randomly
            self.positions = np.where(
                (np.random.random((self.num_bats,)) >= self.pulse_rate)[:, np.newaxis],
                self.best_position[np.newaxis, :]
                + np.random.random((self.num_bats, *self.observation_shape))
                * np.mean(self.loudness),
                self.positions,
            )

            self.fitness_values = self.objective(self.positions)

            condition: np.ndarray = (
                np.random.random((self.num_bats,)) <= self.loudness
            ) & (self.fitness_values > self.best_value)
            self.loudness = np.where(
                condition, self.alpha * self.loudness, self.loudness
            )
            self.pulse_rate = np.where(
                condition,
                self.initial_pulse_rate * (1 - np.exp(-self.gamma * i)),
                self.pulse_rate,
            )
            self.frequencies = self.min_frequency + (
                self.max_frequency - self.min_frequency
            ) * np.random.random((self.num_bats,))

            self.update_best()

            if (i + 1) % 50 == 0:
                print(
                    "iteration {} finished, round's best is {} and average score of this round is {}".format(
                        i + 1,
                        np.round(np.max(self.fitness_values), 3),
                        np.round(np.mean(self.fitness_values), 3),
                    )
                )

                if save_path != None:
                    np.save(save_path, self.best_position)

        return self.best_position

import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List
from math import inf


class PSO:
    def __init__(
        self,
        observation_shape: List[int],
        objective: Callable[[np.ndarray], float],
        min_velocity: float,
        max_velocity: float,
        num_iterations: int = 100,
        c_1: float = 1,
        c_2: float = 1,
        c_3: float = 1,
        w_start: float = 0.5,
        w_end: float = 0.1,
        random_factor: float = 0.5,
    ) -> None:
        self.objective: Callable[[np.ndarray], np.ndarray] = objective
        self.min_velocity: float = min_velocity
        self.max_velocity: float = max_velocity
        self.agent_positions: np.ndarray = np.empty(observation_shape)
        self.agent_velocities: np.ndarray = np.empty(observation_shape)
        self.pbests: np.ndarray = np.empty(observation_shape)
        self.pbest_values: np.ndarray = np.empty((observation_shape[0],))
        self.num_iterations: int = num_iterations
        self.c_1: float = c_1
        self.c_2: float = c_2
        self.c_3: float = c_3
        self.w_start: float = w_start
        self.w_end: float = w_end
        self.random_factor: float = random_factor
        self.gbest: np.ndarray = None
        self.gbest_value: float = -inf
        self.rbest: np.ndarray = None
        self.plot_data_avg: np.ndarray = np.empty((num_iterations,))
        self.plot_data_best: np.ndarray = np.empty((num_iterations,))

    def update_gbest(self):
        new_best_index = np.argmax(self.pbest_values)
        new_best = self.pbest_values[new_best_index]
        if new_best > self.gbest_value or self.gbest is None:
            self.gbest = self.pbests[new_best_index]
            self.gbest_value = new_best
        self.rbest = self.pbests[new_best_index]

    def run(self, inital_positions: np.ndarray, save_path=None) -> np.ndarray:
        self.gbest: np.ndarray = None
        self.gbest_value: float = -inf

        self.agent_positions = inital_positions
        self.agent_velocities = np.zeros(self.agent_positions.shape)
        self.pbests = inital_positions
        self.pbest_values = self.objective(inital_positions)
        self.update_gbest()
        for k in range(self.num_iterations):
            w: float = self.w_start - (self.w_start - self.w_end) * k / (
                self.num_iterations - 1
            )

            self.agent_velocities = np.clip(
                w * self.agent_velocities
                + self.c_1
                * np.random.random((self.agent_positions.shape[0],))[:, np.newaxis]
                * (self.pbests - self.agent_positions)
                + self.c_2
                * np.random.random((self.agent_positions.shape[0],))[:, np.newaxis]
                * (self.rbest - self.agent_positions)
                + self.c_3
                * np.random.random((self.agent_positions.shape[0],))[:, np.newaxis]
                * (self.gbest - self.agent_positions)
                + self.random_factor * np.random.random((self.agent_velocities.shape)),
                self.min_velocity,
                self.max_velocity,
            )
            norm_velocity = np.linalg.norm(self.agent_velocities, axis=1)
            self.agent_velocities = np.where(
                (norm_velocity < self.min_velocity)[:, np.newaxis],
                self.min_velocity
                * (self.agent_velocities / norm_velocity[:, np.newaxis]),
                self.agent_velocities,
            )
            self.agent_velocities = np.where(
                (norm_velocity > self.max_velocity)[:, np.newaxis],
                self.max_velocity
                * (self.agent_velocities / norm_velocity[:, np.newaxis]),
                self.agent_velocities,
            )

            self.agent_positions = self.agent_positions + self.agent_velocities
            new_scores: np.ndarray = self.objective(self.agent_positions)
            self.plot_data_avg[k] = np.mean(new_scores)
            self.plot_data_best[k] = np.max(new_scores)
            condition: np.ndarray = new_scores > self.pbest_values
            self.pbest_values = np.where(condition, new_scores, self.pbest_values)
            self.pbests = np.where(
                condition[:, np.newaxis], self.agent_positions, self.pbests
            )
            self.update_gbest()

            if (k + 1) % 50 == 0:
                print(
                    "iteration {} finished, round's best is {} and average score of this round is {}".format(
                        k + 1,
                        np.round(np.max(new_scores), 3),
                        np.round(np.mean(new_scores), 3),
                    )
                )
                if save_path != None:
                    np.save(save_path, self.gbest)

        np.save(save_path, self.gbest)

        plt.plot(self.plot_data_best, label="best score")
        plt.plot(self.plot_data_avg, label="avg score")
        plt.title("Score of population over iterations")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.show()

        return self.gbest

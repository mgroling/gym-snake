import numpy as np
import matplotlib.pyplot as plt

from typing import Callable, List


class ABC:
    def __init__(
        self,
        num_cycles: int,
        observation_shape: List[int],
        objective: Callable[[np.ndarray], float],
        colony_size: int = 10,
        scout_limit_start: int = 10,
        scout_limit_end: int = 100,
    ) -> None:
        self.num_cycles: int = num_cycles
        self.observation_shape: np.ndarray = observation_shape
        self.objective: Callable[[np.ndarray], float] = objective
        self.colony_size: int = colony_size
        self.scout_limit_start: int = scout_limit_start
        self.scout_limit_end: int = scout_limit_end
        self.scout_limit: int = scout_limit_start
        self.food_sources = (
            np.random.random((colony_size // 2, *observation_shape)) * 2 - 1
        )
        self.improved_solution = np.zeros((len(self.food_sources),), dtype=bool)
        self.scout_count = np.zeros((len(self.food_sources),))
        self.fitness_values = self.fitness(self.food_sources)
        self.best_solution = np.zeros(observation_shape)
        self.best_solution_fitness = self.fitness(self.best_solution[np.newaxis, :])
        self.plot_data_avg: np.ndarray = np.empty((num_cycles,))
        self.plot_data_best: np.ndarray = np.empty((num_cycles,))

    def fitness(self, food: np.ndarray):
        temp_fitness = self.objective(food)
        return np.where(
            temp_fitness > 0, 1 / (1 + temp_fitness), 1 + np.abs(temp_fitness)
        )

    def inverse_fitness(self, fitness_value: np.ndarray):
        return -np.where(
            self.fitness_values < 1,
            1 / self.fitness_values - 1,
            1 - self.fitness_values,
        )

    def employed_bees_phase(self):
        # for each food source select another one
        selected_food_sources = np.random.randint(
            len(self.food_sources) - 1, size=(len(self.food_sources),)
        )
        selected_food_sources = np.where(
            selected_food_sources == np.arange(len(self.food_sources)),
            len(self.food_sources) - 1,
            selected_food_sources,
        )

        # for each food source select a dimension
        selected_dimension = np.random.randint(
            0, self.observation_shape, size=(len(self.food_sources),)
        )

        # explore a new food source along the dimension
        new_potential_food_sources = np.empty_like(self.food_sources)
        for i in range(len(self.food_sources)):
            new_potential_food_sources[i] = self.food_sources[i] + (
                np.random.random() * 2 - 1
            ) * (
                self.food_sources[i, selected_dimension[i]]
                - self.food_sources[selected_food_sources[i], selected_dimension[i]]
            )

        # calculate fitness value and greedily select the better one
        new_fitness_values = self.fitness(new_potential_food_sources)
        is_better = new_fitness_values > self.fitness_values
        self.fitness_values = np.where(
            is_better, new_fitness_values, self.fitness_values
        )
        self.food_sources = np.where(
            is_better[:, np.newaxis], new_potential_food_sources, self.food_sources
        )
        self.improved_solution = np.where(is_better, is_better, self.improved_solution)

        return new_fitness_values

    def onlooker_bees_phase(self):
        # select some of the food sources depending on their fitness
        probabilities = self.fitness_values / np.sum(self.fitness_values)
        visited_food_sources = np.random.choice(
            np.arange(len(self.food_sources)),
            size=len(self.food_sources),
            p=probabilities,
        )
        # for each food source select another one
        selected_food_sources = np.random.randint(
            len(self.food_sources) - 1, size=(len(self.food_sources),)
        )
        selected_food_sources = np.where(
            selected_food_sources == visited_food_sources,
            len(self.food_sources) - 1,
            selected_food_sources,
        )

        # for each food source select a dimension
        selected_dimension = np.random.randint(
            0, self.observation_shape, size=(len(self.food_sources),)
        )

        # explore a new food source along the dimension
        new_potential_food_sources = np.empty_like(self.food_sources)
        for i in range(len(self.food_sources)):
            new_potential_food_sources[i] = self.food_sources[
                visited_food_sources[i]
            ] + (np.random.random() * 2 - 1) * (
                self.food_sources[visited_food_sources[i], selected_dimension[i]]
                - self.food_sources[selected_food_sources[i], selected_dimension[i]]
            )

        # calculate fitness value and greedily select the better one
        new_fitness_values = self.fitness(new_potential_food_sources)
        for i in range(len(self.food_sources)):
            if new_fitness_values[i] > self.fitness_values[visited_food_sources[i]]:
                self.fitness_values[visited_food_sources[i]] = new_fitness_values[i]
                self.food_sources[visited_food_sources[i]] = new_potential_food_sources[
                    i
                ]
                self.improved_solution[visited_food_sources[i]] = True

        return new_fitness_values

    def scout_bee_phase(self):
        # memorize best solution
        current_best = np.argmax(self.fitness_values)
        if self.fitness_values[current_best] > self.best_solution_fitness:
            self.best_solution_fitness = self.fitness_values[current_best]
            self.best_solution = self.food_sources[current_best]

        # increase scout counter
        self.scout_count = np.where(self.improved_solution, 0, self.scout_count + 1)
        self.improved_solution = np.zeros_like(self.improved_solution, dtype=bool)

        # check if scout counter is higher than limit and if yes generate a new random solution
        for i in range(len(self.scout_count)):
            if self.scout_count[i] > self.scout_limit:
                self.scout_count[i] = 0
                self.food_sources[i] = np.random.random(self.observation_shape) * 2 - 1
                self.fitness_values[i] = self.fitness(self.food_sources[np.newaxis, i])

    def run(self, save_path=None) -> np.ndarray:
        self.food_sources = (
            np.random.random((self.colony_size // 2, *self.observation_shape)) * 2 - 1
        )
        self.scout_count = np.zeros((len(self.food_sources),))
        self.fitness_values = self.fitness(self.food_sources)

        for i in range(self.num_cycles):
            self.scout_limit = (
                self.scout_limit_start
                + i * (self.scout_limit_end - self.scout_limit_start) // self.num_cycles
            )
            temp_fitness1 = self.employed_bees_phase()
            temp_fitness2 = self.onlooker_bees_phase()
            self.scout_bee_phase()

            # save data for plotting
            temp = self.inverse_fitness(np.append(temp_fitness1, temp_fitness2))
            self.plot_data_avg[i] = np.mean(temp)
            self.plot_data_best[i] = np.max(temp)

            if (i + 1) % 50 == 0:
                print(
                    "iteration {} finished, round's best fitness is {} and average fitness of this round is {}".format(
                        i + 1,
                        np.round(np.max(temp), 3),
                        np.round(np.mean(temp), 3),
                    )
                )
                if save_path != None:
                    np.save(save_path, self.best_solution)

        plt.plot(self.plot_data_best, label="best score")
        plt.plot(self.plot_data_avg, label="avg score")
        plt.title("Score of population over iterations")
        plt.legend()
        plt.xlabel("Iteration")
        plt.ylabel("Score")
        plt.show()

        return self.best_solution

"""
(C) 2022: Hans Georg Schaathun <hasc@ntnu.no>

A basic, generic, GA implementation for testing and instruction.

from: https://github.com/hgeorgsch/pygax
"""

from ContinuousChromosome import *
import numpy as np
import sys


# Default auxiliary functions
def simpleMutate(x, cprob=0.05, gprob=0.05):
    """
    Mutate the chromosome with probability cprob.
    If the chromosome is mutated, flip each bit with probability gprob.
    """
    if np.random.rand() < cprob:
        l = len(x)
        bc = int(np.ceil(l * gprob))
        ic = np.random.randint(l, size=bc)
        x.flip(ic)
    return x


def simpleMutateFunction(cprob=0.05, gprob=0.05):
    return lambda x: simpleMutate(x, cprob, gprob)


def simpleMate(pop):
    """
    Splits the given population into adjacent pairs, starting at
    the start of the list.
    """
    r = []
    i = 0
    l = len(pop)
    for i in range(0, l - 1, 2):
        r.append((pop[i], pop[i + 1]))
    return r


def cross(x, y):
    beta = np.random.random_sample(size=len(x))
    assert len(x) == len(y)
    r1 = beta * x.gene + (1 - beta) * y.gene
    r2 = beta * y.gene + (1 - beta) * x.gene
    return (ContinuousChromosome(r1), ContinuousChromosome(r2))


# The GA proper
class GA:
    def __len__(self):
        if self.population == None:
            return 0
        else:
            return len(self.population)

    def __init__(
        self,
        representation,
        costfunction,
        mate=simpleMate,
        crossover=cross,
        mutate=simpleMutate,
        selectionrate=0.5,
        debug=2,
    ):
        """
        Initialise the GA with a given representation and cost function.

        Arguments:
            representation : representation object defining the mapping from variables to chromosomes
            costfunction : the costfunction to minimise
            mate : function to select mating pairs
            crossover : the crossover function
            mutate : the mutation function (must work in place)
            selectionrate : the fraction of the population to keep

        Note that if the selection rate is 1 or greater, the population
        is not sorted before mating.

        Hint. To change the mutation rate in the default mutation function,
        you can use lambda functions.  For example,
        `mutate=lambda x : simpleMutate(x,cprob=0.01,gprob=0.1)`.
        """
        self.costfunction = costfunction
        self.representation = representation
        self.mate = mate
        self.crossover = crossover
        self.mutate = mutate
        self.selectionrate = selectionrate
        self.population = None
        self.debug = debug

    def initPopulation(self, size):
        "Initialise a population of the given size."
        self.population = self.representation.makePopulation(size)
        self.generation = 0
        self.populationsize = size
        self._init(size)

    def _init(self, size):
        """
        Initialisations specific to GA
        Making subclasses to implement other population based methods,
        it should not be necessary to override `initPopulation`.
        Overriding this method may be necessary.
        """
        self.Nselect = int(2 * np.ceil(self.selectionrate * size / 2))

    def cost(self, x):
        v = self.representation.getFloat(x)
        c = self.costfunction(v)
        if self.debug > 2:
            print(f"{str(x.gene)} -> {v} -> {c}")
        return c

    def evolve(self, ngen=1):
        """Evolve the given number of generations."""
        for i in range(ngen):
            r = self.nextGeneration()
        return r

    def sortCost(self):
        """
        Calculate costs and sort chromosome.
        The return value is a list of cost/chromosome pairs.
        """
        cost = [(self.cost(x), x) for x in self.population]
        cost.sort(key=lambda x: x[0])
        return cost

    def getSolution(self):
        """
        Get the best solution as a triplet (cost,solution,chromosome).
        """
        costlist = self.sortCost()
        (cost, chromosome) = costlist[0]
        solution = self.representation.getFloat(chromosome)
        return (cost, solution, str(chromosome))

    def nextGeneration(self):
        """Evolve one generation."""

        if self.debug > 0:
            print(
                f"Evolving Generation {self.generation}. Population size {len(self)}.",
                file=sys.stderr,
            )

        # Calculate the cost function; list of cost/chromosome pairs
        cost = [(self.cost(x), x) for x in self.population]

        # Sort and keep the best
        if self.Nselect < len(self):
            cost.sort(key=lambda x: x[0])
            cost = cost[: self.Nselect]
        # Note that we do not sort if we keep all the chromosomes

        if self.debug > 2:
            print(f"Nselect {self.Nselect}.", file=sys.stderr)
        if self.debug > 0:
            print(
                f"Mimimum: {cost[0][0]} - Mean: {sum([elem[0] for elem in cost]) /len(cost)} out of {len(cost)} chromosomes.",
                file=sys.stderr,
            )

        # Make mating pairs
        pairs = self.mate(cost)

        # Make offspring with the crossover function
        offspring = [self.crossover(x, y) for ((cx, x), (cy, y)) in pairs]

        # Update population
        self.population = [x for (y, x) in cost]
        for x in offspring:
            self.population.extend(x)

        # Mutate
        for x in self.population:
            self.mutate(x)

        # Advance the generation count
        self.generation += 1

        return self.population

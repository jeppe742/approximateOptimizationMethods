import numpy as np
from travelingSalesmanProblem import TSP
from copy import deepcopy
import matplotlib.pyplot as plt


class GeneticEvolution(TSP):

    def __init__(self, stations=None, distances=None, population_size=10, mutation_rate=0.1, iterations=1000):
        "Create problem for stations to visit"

        super().__init__(stations, distances)

        self.name = "Genetic Evolution"
        self.population = []
        self.population_size = population_size
        self.population_fitness = [float("inf")]*population_size
        self.mutation_rate = mutation_rate
        self.iterations = iterations

    def update_fitness(self):
        "Updates the cost for the populations"

        fitness = [1/self._get_cost(self.population[i]) for i in range(self.population_size)]

        # normalize the fitness score
        self.population_fitness = [fitness[i]/sum(fitness) for i in range(self.population_size)]

        min_cost = 1/max(fitness)
        if min_cost < self.cost:
            self.cost = min_cost
            self.all_cost.append(min_cost)

            solution_idx = fitness.index(max(fitness))
            self.solution = self.population[solution_idx]
            self.all_solutions.append(self.population[solution_idx])

    def selection(self):
        "Selects two parents. The parents are choosen proportional to their fitness "

        # Select two random indecies, with probability proportional to the fitness
        parents_idx = np.random.choice(self.population_size, 2, p=self.population_fitness, replace=False)

        parents = [self.population[i] for i in parents_idx]
        return parents

    def kill(self):
        # Get the cost from the fitness
        cost = [1/self.population_fitness[i] for i in range(self.population_size)]

        for i in range(2):
            # Normalize the cost
            cost = [cost[i] / sum(cost) for i in range(self.population_size - i)]
            # Pick random solution, with probability proportional to its cost
            kill_idx = np.random.choice(self.population_size - i, 1, p=cost, replace=False).item()
            self.population.pop(kill_idx)
            cost.pop(kill_idx)

    def crossover(self, parents):
        # Create the children as copies of the parents
        children = deepcopy(parents)
        # Select where to choose our crossover points
        crossover_points = sorted(np.random.choice(self.n_stations - 1, 2) + 1)

        for i in range(2):
            # get the remaining stations from the second parent, which are not in the crossover section of the first parent
            #remaning_stations = [station for station in parents[1-i][1:-1] if station not in set(children[i][crossover_points[0]:crossover_points[1]])]
            remaning_stations = set(parents[1-i][1:-1]) - set(children[i][crossover_points[0]:crossover_points[1]])
            for j, remaning_station in enumerate(remaning_stations):
                # begin from the second crossover point, and wrap around the list
                index = (j+crossover_points[1]) % (self.n_stations)
                # if we have wrapped around the list, add one to avoid changing the first element
                if index < crossover_points[0]:
                    index += 1
                children[i][index] = remaning_station

            if np.random.random() < self.mutation_rate:
                children[i] = self.mutate(children[i])
        return children

    def initialize(self):

        for _ in range(self.population_size):
            # let's start at station 1
            solution = [1]
            # Create the remaining route, as a random solution
            solution.extend(np.random.permutation(
                np.arange(2, self.n_stations + 1)))
            # return to station 1
            solution.append(1)
            # Convert to numpy array
            solution = np.asarray(solution)

            self.population.append(solution)

        self.update_fitness()

    def mutate(self, child):
        "mutates the genes in the children"
        # Create a copy of the solution, to avoid references
        child_tmp = child.copy()

        # select two indecies to permute
        i, j = np.random.randint(1, self.n_stations, size=2)

        # Swap indecies i and j
        c_tmp = child_tmp[i]
        child_tmp[i] = child_tmp[j]
        child_tmp[j] = c_tmp

        return child_tmp

    def solve(self):
        ""

        self.initialize()

        for i in range(self.iterations):

            parents = self.selection()
            children = self.crossover(parents)
            # add the children to the population
            self.population.extend(children)
            # kill two from the population, to have a constant population size
            self.kill()
            self.update_fitness()


if __name__ == "__main__":
    stations = np.loadtxt('data/dantzig42_stations.txt', delimiter=',')
    distances = np.loadtxt('data/dantzig42_distances.txt', delimiter=',')

    ge = GeneticEvolution(stations=stations, distances=distances)
    ge.solve()
    ge.plot()

from travelingSalesmanProblem import TSP
import numpy as np
import matplotlib.pyplot as plt


class SimulatedAnneiling(TSP):
    """
    Class for solving the Traveling Salesman Problem using Simulated annealing
    https://doi.org/10.1126%2Fscience.220.4598.671
    """

    def __init__(self, stations=None, n_max=5000):
        "Create problem for stations to visit"

        super().__init__(stations)

        self.name = "Simulated Anneiling"
        self.n_max = n_max

    def _permute(self, solution):
        "Returns the solution, where two indicies has been randomly permuted"

        # Create a copy of the solution, to avoid references
        S = solution.copy()

        # select two indecies to permute
        i, j = np.random.randint(1, self.n_stations, size=2)

        # Swap indecies i and j
        S_tmp = S[i]
        S[i] = S[j]
        S[j] = S_tmp

        return S

    def solve(self, steps=False):
        "Solves the traveling salesman problem using simulated anneiling. steps=True returns all iterations"

        # let's start at station 1
        self.solution = [1]
        # Create the remaining route, as a random solution
        self.solution.extend(np.random.permutation(
            np.arange(2, self.n_stations + 1)))
        # return to station 1
        self.solution.append(1)
        # Convert to numpy array
        self.solution = np.asarray(self.solution)

        self.cost = self._get_cost(self.solution)

        # Run the anneiling for n_max iterations
        for k in range(self.n_max):

            update = False

            # Update "temperature"
            #T = 1/np.sqrt(1+k)
            T = 10*0.99**k
            # Generate new solution candidate, by permuting the currest best
            candidate = self._permute(self.solution)

            cost_candidate = self._get_cost(candidate)

            if cost_candidate < self.cost:
                update = True

            else:
                # Calculate the new acceptance rate
                a = np.exp(-(cost_candidate-self.cost)/T)
                u = np.random.random()

                # accept the candidate with probability a
                if u < a:
                    update = True

            if update:
                self.solution = candidate
                self.cost = cost_candidate
                if steps:
                    self.all_cost.append(self.cost)
                    self.all_solutions.append(self.solution)

        return self.solution, self.cost


if __name__ == "__main__":
    s = SimulatedAnneiling(stations=10*np.random.random((15, 2)))
    s.solve(steps=True)
    s.plot(animate=True)

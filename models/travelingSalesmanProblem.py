import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.spatial.distance import cdist


class TSP:
    "Main class for the Traveling Salesman Problem"

    def __init__(self, stations=None, distances=None):
        """
        Initialize the object with stations, that should be visited.

        Args:
            stations (list of tuples / np array) : x and y coordinates of the stations 
            distances (np ndarray): Distances between stations. Defaults to euclidian 
        """

        if stations is not None:
            self.stations = stations
        else:
            self.stations = [[np.cos(pi), np.sin(pi)] for pi in np.linspace(-np.pi, np.pi*3/4, 8)]

        # Convert list to numpy array
        if isinstance(self.stations, list):
            self.stations = np.asarray(self.stations)

        self.n_stations = len(self.stations)
        self.solution = []
        self.all_solutions = []
        self.all_cost = []
        self.cost = float("inf")
        self.name = None

        # calculate the euclidean norm if nothing is given
        if distances is not None:
            self.distances = distances
        else:
            self.distances = cdist(self.stations, self.stations)

    def plot(self, animate=True):
        "Visualize the problem, and solution if available"

        fig, ax = plt.subplots()
        plt.scatter(self.stations[:, 0], self.stations[:, 1])
        plt.title(
            f"Traveling Salesman Problem. model:{self.name}  cost:{self.cost:.2f}")
        plt.xlabel('x')
        plt.ylabel('y')

        # Check if the object contains a solution
        if len(self.solution) > 0:
            # select the correct order to visit stations
            station_route = self.stations[self.solution-1]
            X = station_route[:-1, 0]
            Y = station_route[:-1, 1]
            U = station_route[1:, 0] - station_route[:-1, 0]
            V = station_route[1:, 1] - station_route[:-1, 1]
            quiver = plt.quiver(X, Y, U, V, scale_units='xy', angles='xy', scale=1)
        if animate:
            if len(self.all_cost) == 0:
                raise RuntimeError(
                    'You need to run solve with steps=True before you can create animations')

            def update_quiver(i, quiver, all_solutions, all_cost):
                station_route = self.stations[all_solutions[i]-1]
                X = station_route[:-1, 0]
                Y = station_route[:-1, 1]
                U = station_route[1:, 0] - station_route[:-1, 0]
                V = station_route[1:, 1] - station_route[:-1, 1]

                quiver.set_offsets(np.vstack((X, Y)).T)
                quiver.set_UVC(U, V)
                ax.set_title(
                    f"Traveling Salesman Problem. model:{self.name}  cost:{all_cost[i]:.2f}")
                return quiver

            anim = animation.FuncAnimation(fig, update_quiver,
                                           fargs=(quiver,
                                                  self.all_solutions,
                                                  self.all_cost),
                                           interval=50,
                                           blit=False,
                                           frames=len(self.all_cost),
                                           repeat_delay=3000)
            anim.save(f'gifs/{self.name.replace(" ","_")}.gif', fps=10,
                      writer='imagemagick')
        plt.show()

    def animate(self):
        "Creates an animation of the solution iterations"

        plt.figure()
        plt.scatter(self.stations[:, 0], self.stations[:, 1])
        plt.title(
            f"Traveling Salesman Problem. model:{self.name}  cost:{self.cost:.2f}")
        plt.xlabel('x')
        plt.ylabel('y')

    def _get_cost(self, solution):
        "Returns the cost of the current solution"
        cost = 0
        for i in range(self.n_stations):
            cost += self.distances[solution[i] - 1, solution[i + 1] - 1]
        return cost

    def solve(self):
        raise NotImplementedError(
            "The solver method for this class hasn't been implemented yet")

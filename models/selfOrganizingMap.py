import numpy as np
from travelingSalesmanProblem import TSP
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.stats import norm

# np.seterr(all='raise')


class SelfOrganizingMap(TSP):

    def __init__(self, stations=None, distances=None, iterations=1000, neighborhood=None, neighborhood_decay=0.99, learning_rate=0.8, learning_rate_decay=0.99):
        "Create problem for stations to visit"

        super().__init__(stations, distances)

        self.name = "Self-organizing map"
        self.iterations = iterations

        if neighborhood is None:
            self.neighborhood = self.n_stations
        else:
            self.neighborhood = neighborhood

        self.neighborhood_decay = neighborhood_decay
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def distance(self, station, neurons):
        "returns the euclidean distance between a station and all neurons "

        return np.sqrt((station[0]-neurons[:, 0])**2 + (station[1]-neurons[:, 1])**2)

    def gaussian_kernel(self, x, width):
        return np.exp(-x**2/(2*width**2))

    def solve(self):
        n_neurons = self.n_stations*4 + 1
        # initialize all neurons with random x,y coordinates
        self.neurons = np.random.rand(n_neurons, 2)*10

        # create a pointer from the neuron array to solution, in order to keep the naming convention from the other methods
        self.solution = self.neurons

        for i in range(self.iterations):
            # Look at one station at a time
            for station in self.stations:
                # Find the neuron closest to the station
                winner_neuron = np.argmin(self.distance(station, self.neurons))

                # spread of gaussian kernel
                sigma = max(1, self.neighborhood*self.neighborhood_decay**i)

                # create binary vector to represent if a neuron is in the neighbourhood of the winner neuron
                distance = abs(winner_neuron - np.arange(n_neurons))
                # Calculate the circular distance
                distance = np.minimum(distance, n_neurons - distance)

                neighborhood = self.gaussian_kernel(distance, sigma)

                # stack the binary vector into two columns, so we can multiply it with the coordinates when we do the update
                neighborhood = neighborhood.reshape(n_neurons, 1)
                neighborhood = np.hstack((neighborhood, neighborhood))

                # Update the neurons
                self.neurons += (self.learning_rate*self.learning_rate_decay**i) * neighborhood * (station - self.neurons)

                # l1.set_data(self.neurons[:, 0], self.neurons[:, 1])
            if i % 10 == 0:
                self.all_solutions.append(self.solution.copy())

    def plot(self, animate=True):
        "Visualize the problem, and solution if available"

        fig, _ = plt.subplots()
        plt.scatter(self.stations[:, 0], self.stations[:, 1])
        plt.title(
            f"Traveling Salesman Problem. model:{self.name}")
        plt.xlabel('x')
        plt.ylabel('y')

        # Check if the object contains a solution
        if len(self.solution) > 0:
            line, = plt.plot(self.neurons[:, 0], self.neurons[:, 1], 'r')
        if animate:
            if len(self.all_solutions) == 0:
                raise RuntimeError(
                    'You need to run solve with steps=True before you can create animations')

            def update_line(i, line, all_solutions):
                neurons = all_solutions[i]

                line.set_data(neurons[:, 0], neurons[:, 1])

                return line

            anim = animation.FuncAnimation(fig, update_line,
                                           fargs=(line,
                                                  self.all_solutions),
                                           interval=50,
                                           blit=False,
                                           frames=len(self.all_solutions),
                                           repeat_delay=3000)
            anim.save(f'gifs/{self.name.replace(" ","_")}.gif', fps=10,
                      writer='imagemagick')
        plt.show()


if __name__ == "__main__":
    stations = np.loadtxt('data/dantzig42_stations.txt', delimiter=',')
    distances = np.loadtxt('data/dantzig42_distances.txt', delimiter=',')

    som = SelfOrganizingMap(stations=stations, distances=distances)
    som.solve()
    som.plot()

import numpy
import random
import matplotlib.pyplot


####### CONFIG #######
k       = 7
in_file = 'test_1.txt'
######################


def sqrnorm(p1: list, p2: list) -> float:
        """ Square norm: ||x1 - x2||^2. """
        return numpy.linalg.norm(numpy.array(p1) - numpy.array(p2)) ** 2


def minnorm_index(centroids: list, point: list):
        """ Index of centroid with minimum norm from {||x-uj||^2 for uj in centroids}. """
        return min(range(len(centroids)), key = lambda cent_ind: sqrnorm(point, centroids[cent_ind]))


def set_row_index(row: list, index: int) -> list:
        """ Sets every element on the row to 0 except the one at index, which is 1. """
        _row = [0 for _ in row]
        _row[index] = 1
        return _row


def compute_j_score(gamma: list, points: list, centroids: list) -> float:
        """ Given the matrix gamma, the list of centroids and the list of points, returns the j-score. """
        return sum([gamma[i][j] * sqrnorm(points[i], centroids[j]) for i in range(len(gamma)) for j in range(len(gamma[0]))])


def choose_n(n: int, l: list) -> list:
        """ Randomly sort the list l and then return the first n elements. """
        return sorted(l, key = lambda _: random.random())[:n]


class cluster_of:
        @staticmethod
        def from_gamma(cls_ind: int, gamma: list) -> list:
                """ Returns a list indices of points representing the j-th cluster. """
                return [i for i in range(len(gamma)) if gamma[i][cls_ind] == 1]

        @staticmethod
        def from_lists(cls_ind: int, centroids: list, points: list):
                """ Returns a list of indices of the points belonging to the j-th cluster. """
                return [point_ind for point_ind in range(len(points)) if minnorm_index(centroids, points[point_ind]) == cls_ind]


class clusterize:
        @staticmethod
        def from_gamma(gamma: list, centroids: list, points: list) -> list:
                """ Returns a new gamma matrix with the clusters for the given centroids using the previous gamma matrix. (faster) """
                return [set_row_index(gamma[i], j) for j in range(len(centroids)) for i in cluster_of.from_gamma(j, gamma)]

        @staticmethod
        def from_lists(centroids: list, points: list) -> list:
                """ Returns a new gamma matrix with the clusters for the given centroids recomputing the distances from each point to each centroid. (slower) """
                return [set_row_index(centroids, j) for j in range(len(centroids)) for i in cluster_of.from_lists(j, centroids, points)]


def compute_centorids(gamma: list, points: list) -> list:
        """ For each centroid get the indices of the points of its cltuster from gamma, then compute the sums of the coordinates and divide the resulting matrix. """
        centroids = []
        for cls_ind in range(len(gamma[0])):
                new_centroid = (sum([gamma[pnt_ind][cls_ind] * numpy.array(points[pnt_ind]) for pnt_ind in range(len(gamma))])
                                / sum([gamma[pnt_ind][cls_ind] for pnt_ind in range(len(gamma))]))
                centroids.append(new_centroid)
        return centroids


def plot_2d_clustering(gamma: list, centroids: list, points: list):
        """ Plot an R^2 graph with up to 7 different clusters represented as different colors. """
        colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
        for cls_ind in range(min(len(gamma[0]), len(colors))):
                xs, ys = zip(*[points[ind] for ind in cluster_of.from_lists(cls_ind, centroids, points)])
                matplotlib.pyplot.plot(xs, ys, colors[cls_ind])
        matplotlib.pyplot.ylabel('Y-axis')
        matplotlib.pyplot.xlabel('X-axis')
        matplotlib.pyplot.show()


def plot_j_values(j_values: list):
        matplotlib.pyplot.plot(list(range(0, len(j_values))), j_values)
        matplotlib.pyplot.ylabel('J value')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.show()


j_values = []                                                                                           # Initial J value list is empty.
points = [[int(coord) for coord in line.split(' ')] for line in open(in_file, 'r')]                     # Initial data points.

centroids = choose_n(k, points)                                                                         # Compute initial k random centroids.
gamma = clusterize.from_lists(centroids, points)                                                        # Compute initial clusters.
j_values.append(compute_j_score(gamma, points, centroids))                                              # Compute initial J score.
print(f'Iteration   0   J = {j_values[-1]:8.3f}')

while True:
        centroids = compute_centorids(gamma, points)                                                    # Compute next iteration centroids.
        gamma = clusterize.from_lists(centroids, points)                                                # Compute next iteration clusters.
        j_values.append(compute_j_score(gamma, points, centroids))                                      # Compute next iteration J score.
        print(f'Iteration {len(j_values):3}   J = {j_values[-1]:8.3f}')
        if not j_values[-1] < j_values[-2]:
                break

plot_j_values(j_values)
plot_2d_clustering(gamma, centroids, points)

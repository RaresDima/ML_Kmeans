import numpy
import matplotlib.pyplot
from random import choice
from random import randint
from random import uniform
from typing import Any
from typing import Sequence
from typing import Iterable
from typing import Callable
from decimal import Decimal
from decimal import getcontext
from itertools import islice
from pprint import pprint as _pprint

getcontext().prec = 6


def pprint(purpose: str, *x: object):
        """ Print the purpose string, a line of '-' as long as the purpose string and then pprint the argument. """
        print(purpose)
        print('-' * len(purpose))
        _pprint(*x)
        print()


def avg(*args: Any):
        """ Returns the average value of the arguments. Arguments must implement (+) on objects of the same type and (/) on int objects"""
        return sum(args) / len(args)


def head(seq: Sequence) -> Any:
        """ Returns the first element of the iterable argument """
        return seq[0]


def takefirst(iterable: Iterable) -> Any:
        """ Similar to head but also works on iterators and generators as well as sequence types. """
        return head(tuple(islice(iterable, 1)))


def takefirst_such_shat(predicate: Callable, iterable: Iterable) -> Any:
        """ Return the first element in the iterable that satisfies the predicate. """
        return takefirst(filter(predicate, iterable))


def euclidean(pair: Sequence[Iterable]) -> Decimal:
        """ Returns a Decimal object representing the euclidean distance between the 2 points in pair. """
        return Decimal(numpy.linalg.norm(numpy.array(pair[0]) - numpy.array(pair[1])))


def inverse_euclidean(pair: Sequence[Iterable]) -> Decimal:
        """ One over euclidean distance. Used to reverse the monotony of euclidean. """
        return Decimal(1 / euclidean(pair))


def distance_to(distance_func: Callable, point: Iterable) -> Callable:
        """ Returns a new function that computes takes only one argument(other point), a point, and returns the distance from that other point to the main point. """
        return lambda other_point: distance_func((point, other_point))


def sqr_mean(p1: Iterable, p2: Iterable) -> float:
        """ Returns the square mean of the 2 points: ||p1 - p2||^2. """
        return sum([(u - v) ** 2 for u, v in zip(p1, p2)])


def center_of_mass(point_iterable: Iterable) -> list:
        """ Returns the point that represents the center of mass for the given iterable of points. """
        point_iterable = tuple(point_iterable)
        points         = len(point_iterable)
        dimensions     = len(head(point_iterable))
        component_avg  = [Decimal(0)] * dimensions
        for component_index in range(dimensions):
                for point in point_iterable:
                        component_avg[component_index] += Decimal(point[component_index])
                component_avg[component_index] /= Decimal(points)
        return component_avg


def choose_n(n: int, iterable: Iterable) -> Any:
        """ Returns n objects from the given iterable without replacement. """
        iterable = tuple(iterable)
        element  = 0
        result   = []
        for i in range(n):
                while iterable[element] in result:
                        element = randint(0,len(iterable))
                result.append(iterable[element])
        return result


def kmeanspp_init(distance_func: Callable, n: int, iterable: Iterable) -> list:
        """ Implements a rudimentary K-Means++ initialization, choosing n initial centroids and returning them as a list. """
        iterable   = list(iterable)
        result     = []
        init_point = choice(iterable)
        iterable.remove(init_point)
        result.append(init_point)
        while len(result) < n:
                dx = list(map(lambda element: Decimal(distance_to(distance_func, init_point)(element) ** 2), iterable))
                probability_range = float(sum(dx))
                selector = uniform(0.0, probability_range)
                for index, distance in enumerate(dx):
                        if selector >= sum(dx[:index]):
                                result.append(iterable[index])
                                iterable.remove(iterable[index])
                                break
        return result


def argmax(fitness: Callable, iterable: Iterable) -> Any:
        """ Implements argmax. Given an iterable and a fitness function over that iterable returns the element from the iterable that yielded the highest fitness. """
        iterable     = tuple(iterable)
        fitness_list = tuple(map(lambda element: (fitness(element), element), iterable))
        max_fitness  = max(fitness_list, key=head)[0]
        return takefirst_such_shat(lambda element: fitness(element) == max_fitness, iterable)


def argmin(fitness: Callable, iterable: Iterable) -> Any:
        """ Implements argmin. Given an iterable and a fitness function over that iterable returns the element from the iterable that yielded the lowest fitness. """
        iterable     = tuple(iterable)
        fitness_list = tuple(map(lambda element: (fitness(element), element), iterable))
        min_fitness  = min(fitness_list, key=head)[0]
        return takefirst_such_shat(lambda element: fitness(element) == min_fitness, iterable)


def kmeans_clusterize(distance_func: Callable, centroid_iterable: Iterable, point_iterable: Iterable) -> dict:
        """ Given an iterable of centroids and one of points returns a dict where the key for each cluster is the name(str) of the centroid(C1, C2, C3, etc.). """
        clusters = {centroid.name : [] for centroid in centroid_iterable}
        for point in point_iterable:
                nearest_centroid = argmin(distance_to(distance_func, point), centroid_iterable)
                clusters[nearest_centroid.name].append(point)
        return clusters


def compute_j_score(centroids: Iterable, clusters: dict) -> float:
        """ Computess the J score for the current clustering using the square mean. """
        j_score = 0
        for centroid in centroids:
                for point in clusters[centroid.name]:
                        j_score += sqr_mean(point, centroid)
        return j_score


def plot_2d_clustering(*flat_forms: list):
        """ Plot an R^2 graph with up to 7 different clusters represented as different colors. """
        colors = ['bo', 'go', 'ro', 'co', 'mo', 'yo', 'ko']
        for i, form in enumerate(flat_forms[:min(len(colors), len(flat_forms))]):
                xs, ys = zip(*form)
                matplotlib.pyplot.plot(xs, ys, colors[i])
        matplotlib.pyplot.ylabel('Y-axis')
        matplotlib.pyplot.xlabel('X-axis')
        matplotlib.pyplot.show()


def plot_j_values(j_values: Sequence):
        """ Given a sequence type of values, plots the values as a graph. """
        matplotlib.pyplot.plot(list(range(0, len(j_values))), j_values)
        matplotlib.pyplot.ylabel('J value')
        matplotlib.pyplot.xlabel('Iteration')
        matplotlib.pyplot.show()

from Types import Point
from Utils import pprint
from Utils import choose_n
from Utils import euclidean
from Utils import kmeanspp_init
from Utils import center_of_mass
from Utils import compute_j_score
from Utils import kmeans_clusterize
from Utils import plot_j_values
from Utils import plot_2d_clustering
from decimal import Decimal
from decimal import getcontext

############ CONFIG ############
k                 = 3
in_file           = 'test_1.txt'
distance_func     = euclidean
precision         = 6
################################

getcontext().prec = precision

j_values = []
data_points = list(map(lambda i_p_pair: Point('X' + str(i_p_pair[0] + 1), i_p_pair[1]),
                       enumerate(map(lambda raw_point_data: map(Decimal, raw_point_data.split()),
                                     open(in_file, 'r').read().splitlines()))))
pprint('INITIAL DATA POINTS:', data_points)

centroids = kmeanspp_init(distance_func, k, data_points)

clusters_old = None
clusters = kmeans_clusterize(distance_func, centroids, data_points)
j_values.append(compute_j_score(centroids=centroids, clusters=clusters))

while clusters != clusters_old:
        centroids = [Point('C' + str(index + 1), center_of_mass(clusters[centroid.name])) for index, centroid in enumerate(centroids)]
        clusters_old, clusters = clusters, kmeans_clusterize(distance_func, centroids, data_points)
        j_values.append(compute_j_score(centroids=centroids, clusters=clusters))
        print(f'J for ITER {str(len(j_values)).rjust(2)} : {str(j_values[-1]).rjust(10)}')

plot_j_values(j_values)
plot_2d_clustering(*clusters.values())

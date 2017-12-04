from igraph import *
import numpy as np
import random

DIRECTORY = 'DATA/model3_'

tmax = 10000

output_t_i = [1, 10, 100, 1000, tmax]

def ba_generation(graph, t_i):
    for i in range(graph.ecount(), t_i):
        random_vertex = random.choice(graph.vs)
        random_vertex2 = random.choice(graph.vs)
        while random_vertex == random_vertex2:
            random_vertex = random.choice(graph.vs)

        graph.add_edge(random_vertex2, random_vertex)

    return graph


G = Graph(tmax)

for t_i in output_t_i:
    print(str(t_i) + " step")
    G = ba_generation(G, t_i+1)
    with open(DIRECTORY + 'at_' + str(t_i) + '.edges', 'w') as f:
        G.write_edgelist(f)


np.savetxt(DIRECTORY + 'hist_degree.txt',
            np.array(list(G.degree_distribution().bins()))[:,[0,2]],
            fmt="%d"
)

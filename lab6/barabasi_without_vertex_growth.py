from igraph import *
import random

FILE = "data/ti_without_vertex_growth_albert.data"

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
    with open(str(t_i) + "_" + FILE, 'w') as f:
        G.write_edgelist(f)

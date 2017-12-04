from igraph import *
import random

FILE = "data/without_preferential_attachment.data"

tmax = 10000

output_t_i = [1, 10, 100, 1000, tmax]

def ba_generation(graph, t_i):
    for i in range(graph.vcount(), t_i):
        random_vertex = random.choice(graph.vs)
        graph.add_vertex(i)
        graph.add_edge(i, random_vertex)

    return graph


G = Graph(1)

for t_i in output_t_i:
    G = ba_generation(G, t_i+1)
    with open(str(t_i) + "_" + FILE, 'w') as f:
        G.write_edgelist(f)

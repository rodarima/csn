from igraph import *
import pandas as pd
import numpy as np
import random


DIRECTORY = 'DATA/model'

tmax = 10000

output_t_i = [1, 10, 100, 1000, tmax]

def model1(t_i, graph = Graph(1)):
    G = Graph.Barabasi(t_i+1, 1, outpref=True, start_from=graph)
    return G

def model2(t_i, graph = Graph(1)):
    for i in range(graph.vcount(), t_i+1):
        random_vertex = random.choice(graph.vs)
        graph.add_vertex(i)
        graph.add_edge(i, random_vertex)

    return graph

def model3(t_i, graph = Graph(tmax)):
    for i in range(graph.ecount(), t_i):
        random_vertex = random.choice(graph.vs)
        random_vertex2 = random.choice(graph.vs)
        while random_vertex == random_vertex2:
            random_vertex = random.choice(graph.vs)

        graph.add_edge(random_vertex2, random_vertex)

    return graph

models = [model1, model2, model3]

def build_table(model, t_i, table, G = None):
    if G == None:
        G = model(t_i)
    else:
        G = model(t_i, G)

    if table.shape[0] == 0:
        table = np.array([
        np.arange(G.vcount()),
        G.degree(),
        np.repeat(t_i, G.vcount())]).T
        table = table.reshape([G.vcount(), 3])
    else:
        rows = np.array([
        np.arange(G.vcount()),
        G.degree(),
        np.repeat(t_i, G.vcount())]).T
        table = np.append(table, rows, axis=0)

    # with open(DIRECTORY + 'at_' + str(t_i) + '.edges', 'w') as f:
        # G.write_edgelist(f)

    return (G,table)


for index_model in range(0,len(models)):
    table = np.array([])
    for t_i in output_t_i:
        print("Model" + str(index_model+1) + ": " + str(t_i) + " step")
        if (t_i == 1):
            G,table = build_table(models[index_model], t_i, table)
        else:
            G,table = build_table(models[index_model], t_i, table, G)
        np.savetxt(DIRECTORY + str(index_model+1) + '_hist_degree.txt',
                    np.array(list(G.degree_distribution().bins()))[:,[0,2]],
                    fmt="%d"
        )

        np.savetxt(DIRECTORY + str(index_model+1) + '_tis_degree.txt',
                    table,
                    fmt="%d"
        )

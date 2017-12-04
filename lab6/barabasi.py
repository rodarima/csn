from igraph import *
import numpy as np
import random

DIRECTORY = "data/model1_"

tmax = 10000

output_t_i = [1, 10, 100, 1000, tmax]

G = Graph(1)

for t_i in output_t_i:
    print(str(t_i) + " step")
    G = Graph.Barabasi(t_i+1, 1, outpref=True, start_from=G)
    with open(DIRECTORY + "at_" + str(t_i) + ".edges", 'w') as f:
        G.write_edgelist(f)

np.savetxt(DIRECTORY + 'hist_degree.txt',
            np.array(list(G.degree_distribution().bins()))[:,[0,2]],
            fmt="%d"
)

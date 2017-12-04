from igraph import *
# from Graph import *

FILE = "ti_albert.data"

tmax = 10000

output_t_i = [10, 100, 1000, tmax]

G = Graph.Barabasi(2,1)
with open("1_" + FILE, 'w') as f:
  G.write_edgelist(f)

for t_i in output_t_i:
  G = Graph.Barabasi(t_i+1, 1, outpref=True, start_from=G)
  with open(str(t_i) + "_" + FILE, 'w') as f:
    G.write_edgelist(f)

# plot(G)

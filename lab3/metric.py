from random import *

seed(64718)
metrics = ["clustering coefficient", "closeness centrality"]
r = randint(0, 1)

print(metrics[r])

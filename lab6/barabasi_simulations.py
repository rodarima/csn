from igraph import *
import pandas as pd
import numpy as np
import random

VERBOSE = 0
TMAX_POWER = 4
tmax = 10**TMAX_POWER
arrival_times = 10 ** np.arange(4)

np.random.seed(1)

data_dir = 'data/'

# dd means degree distribution
data_dd_fmt = 'model{}/dd.txt'

# dseq means degree sequence
data_dseq_fmt = 'model{}/dseq.txt'

# dt means degree over time
data_dt_fmt = 'model{}/dt{}.txt'

def model1(G, steps, m0=1):
	V = G.vcount()
	# Continue adding vertex until we have V + step
	if VERBOSE:
		print('Running barabasi from {} to {}'.format(V, V+steps))
	G = Graph.Barabasi(V + steps, m0, outpref=True, start_from=G)
	return G

def model2(G, steps, m0=1):
	if VERBOSE:
		V = G.vcount()
		print('Running model 2 from {} to {}'.format(V, V+steps))
	for i in range(steps):
		new = G.vcount()
		targets = np.random.choice(G.vs, [m0])
		G.add_vertex(new)
		for to in targets:
			G.add_edge(new, to)

	return G

def model3(G, steps, m0=1):
	if VERBOSE:
		E = G.ecount()
		V = G.vcount()
		print('Running model 3 from E={} to E={}, V={}'.format(E, E+steps*m0, V))
	for i in range(steps):
		# Don't use replacement to avoid loops/multiedges
		selection = np.random.choice(G.vs, [m0+1], replace=False)
		
		# Extract one vertex as the one to be connected
		selected = [selection[0]] * m0
		targets = selection[1:]

		# Connect
		G.add_edges(list(zip(selected, targets)))

	return G

models = [model1, model2, model3]

def next_step(model, G, t, tracing):
	# Advance one step
	steps = 1
	G = model(G, steps)
	if VERBOSE or (t % 1000) == 0:
		print('Time {}: G has {} vertices and {} edges'.format(
			t, G.vcount(), G.ecount()))

	# If we just passed some time_trace[i], add a new tracing vertex
	if t in arrival_times:
		# The added vertex should have index = t
		if VERBOSE:
			print('Adding vertex {} to the tracing vertices'.format(t))
		tracing[t] = []

	# Get the degree distribution
	dd = G.degree()

	# Keep track of the tracing vertices
	for v in tracing.keys():
		tracing[v].append((t, dd[v]))

	# with open(DIRECTORY + 'at_' + str(t_i) + '.edges', 'w') as f:
		# G.write_edgelist(f)

	return G

def save_tracing(name, tracing):
	for v in tracing.keys():
		fn = data_dir + data_dt_fmt.format(name, v)
		table = np.array(tracing[v])
		np.savetxt(fn, table, fmt="%d")

def save_dd(name, dd):
	fn = data_dir + data_dd_fmt.format(name)
	np.savetxt(fn, dd, fmt="%d")

def save_dseq(name, dseq):
	fn = data_dir + data_dseq_fmt.format(name)
	np.savetxt(fn, dseq, fmt="%d")

def simulate_model(m, G0):
	model = models[m]
	name = str(m + 1)
	print('Simulating model {} up to tmax = {}'.format(name, tmax))

	# First build the starting unconnected graph with 1 vertex
	G = G0
	t = 0

	# At start we don't have any tracing vertices
	tracing = {}

	# Continue the simulation until t = tmax
	for t in range(1, tmax + 1):
		G = next_step(model, G, t, tracing)

	dd = np.array(list(G.degree_distribution().bins()))[:,[0,2]]
	dseq = np.array([G.vs.indices, G.degree()]).T

	# Save metrics
	save_tracing(name, tracing)
	save_dd(name, dd)
	save_dseq(name, dseq)
	

def main():
	G0 = [Graph(1), Graph(1), Graph(tmax)]
	for m in range(len(models)):
		simulate_model(m, G0[m])

if __name__ == '__main__':
	main()

import networkx as nx
import numpy as np
import re
from tabulate2 import tabulate

#edges = [(1,2), (1,4), (2,3), (2,4), (3,4)]
#G = nx.Graph(edges)
#
#print(nx.transitivity(G))
#print(nx.average_clustering(G))

LANGUAGES = 'languages.txt'
#LANGUAGES = 'basque.txt'
TABLE2 = 'table2.tex'
GRAPH_FILE = 'data/{}_syntactic_dependency_network.edges'
T = 25

def seq_er(N, M, T=20):
	measure = np.zeros(T)
	for i in range(T):
		print("{:.1f}%".format(100*i/T))
		erG = nx.gnm_random_graph(N, M)
		measure[i] = nx.average_clustering(erG)

	return measure

def model_er(G):
	x = nx.average_clustering(G)
	N = G.order()
	E = G.size()
	measures = seq_er(N, E, T)
	prob = np.sum(measures >= x)/T
	return (prob, x, np.mean(measures))

def table2_row(fn, lang):
	G = nx.read_edgelist(fn, comments='\0', delimiter=' ',
		create_using=nx.Graph(), encoding='utf-8')
	G.remove_edges_from(G.selfloop_edges())
	
	prob_er, x, mean_er = model_er(G)

	row = np.array([lang, x, mean_er, prob_er])
	return row

def table2(langs_fn):
	with open(langs_fn, 'r') as f:
		langs = f.readlines()

	langs = [lang.strip() for lang in langs]
	table = []
	for lang in langs:
		print(lang)
		lang_graph = GRAPH_FILE.format(lang)
		row = table2_row(lang_graph, lang)
		table.append(row)

	latex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=".3E",
		headers=['Language', '$x$', '$\overline x_{ER}$', '$p(x_{ER} \ge x)$'])
	
	latex_table = re.sub('([+-]?[0-9]\.[0-9]*E[+-]?[0-9]*)', r'\\num{\1}', latex_table)
	
	with open(TABLE2, 'w') as f:
		f.write(latex_table)

table2(LANGUAGES)

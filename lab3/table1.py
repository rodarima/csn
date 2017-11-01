import networkx as nx
import numpy as np
import re
from tabulate2 import tabulate

LANGUAGES = 'languages.txt'
TABLE1 = 'table1.tex'
GRAPH_FILE = 'data/{}_syntactic_dependency_network.edges'

def table1_row(fn, lang):
	G = nx.read_edgelist(fn, comments='\0', delimiter=' ',
		create_using=nx.Graph(), encoding='utf-8')
	G.remove_edges_from(G.selfloop_edges())
	N = G.order()
	E = G.size()
	mk = 2*E/N
	delta = 2*E/(N*(N-1))

	row = np.array([lang, N, E, mk, delta])
	return row

def table1(langs_fn):
	with open(langs_fn, 'r') as f:
		langs = f.readlines()

	langs = [lang.strip() for lang in langs]
	table = []
	for lang in langs:
		print(lang)
		lang_graph = GRAPH_FILE.format(lang)
		row = table1_row(lang_graph, lang)
		table.append(row)

	latex_table = tabulate(table,
		tablefmt="latex_booktabs",
		floatfmt=".3E",
		numalign="left",
		headers=['Language', '$N$', '$E$', '$\\langle k \\rangle$', '$\\delta$'])

	latex_table = re.sub('([+-]?[0-9]\.[0-9]*E[+-]?[0-9]*)', r'\\num{\1}', latex_table)
	
	with open(TABLE1, 'w') as f:
		f.write(latex_table)

table1(LANGUAGES)

import networkx as nx
import numpy as np
import re
from tabulate2 import tabulate

CSV = 'table3.csv'
TEX = 'table3.tex'

with open(CSV, 'r') as f:
	data = f.readlines()

table = []
for line in data[1:]:
	row = line.strip().split(',')
	lang = row[1].strip('"')
	x = float(row[2])
	xs = float(row[3])
	prob = float(row[4])
	table.append([lang, x, xs, prob])

latex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=".3f",
	headers=['Language', '$x$', '$\overline x_S$', '$p(x_S \ge x)$'])

latex_table = re.sub('([+-]?[0-9]\.[0-9]*E[+-]?[0-9]*)', r'\\num{\1}', latex_table)

with open(TEX, 'w') as f:
	f.write(latex_table)


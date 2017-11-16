import numpy as np
from tabulate2 import tabulate
import re

LANG_FILE = 'languages.txt'
NUM_FMT = '.3f'
EPSILON = 5e-6 # Avoid low precision round error in the tests

DATA = 'data/{}_dependency_tree_metrics.txt'
TEX = 'table1.tex'

with open(LANG_FILE, 'r') as f:
	LANGS = [e.strip() for e in f.readlines()]

# Random choice has selected degree 2nd moment


# Data test
def test_d2m(n, d2m):
	if np.any((4-6/n) - d2m >= EPSILON): return False
	if np.any(d2m - (n-1) >= EPSILON): return False
	return True

def row1(lang, data):
	N = data.shape[0]
	n = data[:,0]
	d2m = data[:,1]
	if not test_d2m(n, d2m):
		print('Lang {} is not passing the test d2m'.format(lang))
	return [lang, N, np.mean(n), np.std(n), np.mean(d2m), np.std(d2m)]

def table1(langs):
	table = []
	for lang in langs:
		fn = DATA.format(lang)
		data = np.genfromtxt(fn, delimiter=' ')
		row = row1(lang, data)
		table.append(row)
	return table

table = table1(LANGS)
		
latex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=NUM_FMT,
	headers=['Language', '$N$', '$\overline n$', '$s_n$', '$\overline x$', '$s_x$'])

latex_table = re.sub('([+-]?[0-9]\.[0-9]*E[+-]?[0-9]*)', r'\\num{\1}', latex_table)

with open(TEX, 'w') as f:
	f.write(latex_table)

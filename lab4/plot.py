import numpy as np
import pandas as pd
from tabulate2 import tabulate
import re
import matplotlib.pyplot as plt
from scipy.optimize import *
from itertools import cycle
from scipy.stats.stats import pearsonr
import warnings

FIG = 'fig/'
LOSS = 'soft_l1'
VERBOSE = 0
USE_EVOLUTION = False
PLOT = True #True
PLOT_ALL = True #False
SAVE_PLOT = True
METRIC = 'k2'
LANG_FILE = 'languages.txt'
NUM_FMT = '.1f'
EPSILON = 5e-6 # Avoid low precision round error in the tests

# Reproducible runs
np.random.seed(3)

DATA = 'data/{}_dependency_tree_metrics.txt'

with open(LANG_FILE, 'r') as f:
	LANGS = [e.strip() for e in f.readlines()]

#LANGS = ['Czech']

# Reference model
def m0(n): return (1-1/n)*(5-6/n)

# The first models
def m1(n, b):		return (n/2)**b
def m2(n, a, b):	return a*n**b
def m3(n, a, c):	return a*np.exp(c*n)
def m4(n, a):		return a*np.log(n)
def m5(n, a, b, c):	return a * n**b * np.exp(c*n)

# The last models
def m1d(n, b, d):					return (n/2)**b + d
def m2d(n, a=0.06, b=0.8, d=1):		return a*n**b + d
def m3d(n, a=-1, c=-0.05, d=13):	return a*np.exp(c*n) + d
def m4d(n, a, d):					return a*np.log(n) + d
def m5d(n, a, b, c, d):				return a * n**b * np.exp(c*n) + d

bounds = (-15, 14) # Same bounds for all models
models = [m0, m1, m2, m3, m4, m5, m1d, m2d, m3d, m4d, m5d]
models_name = ['0', '1', '2', '3', '4', '5', '1+', '2+', '3+', '4+', '5+']
models_nparam = [0,1,2,2,1,3, 2,3,3,2,4]
models_bounds = [bounds]*len(models)
models_bounds[8] = (-6,16)

def fit_model(data, i):
	model = models[i]
	bounds = models_bounds[i]
	nparam = models_nparam[i]
	n = data.shape[0]
	x = data[:,0]
	y = data[:,1]
	warnings.filterwarnings("ignore")

	if nparam == 0: # Reference model doesn't have params to fit
		popt = []
	else:
		try:
			# Fit model params
			popt, pcov = curve_fit(model, x, y, bounds=bounds, verbose=VERBOSE, loss=LOSS)
		except: # In case of non-convergence try evolution
			def diff_model(p):
				return np.sum((y-model(x, *p))**2)

			print('Fit failed, running differential evolution...')
			de = differential_evolution(diff_model, [bounds]*nparam, seed=1)
			popt = de.x
	
	# Compute residual sum of squares RSS
	yy = model(x, *popt)
	RSS = np.sum((y - yy)**2)

	# Compute Akaike information criterion
	p = 0
	AIC = n*np.log(2*np.pi) + n*np.log(RSS/n) + n + 2*(p + 1)

	# Compute pearson r
	pr, pval = pearsonr(y, yy)

	return (popt, RSS, AIC, pr)

def test_models(data, models):
	results = []
	for i in range(len(models)):
		model = models[i]
		name = models_name[i]

		print("Testing model {}".format(name))
		param, RSS, AIC, pr = fit_model(data, i)
		results.append((name, model, param, RSS, AIC, pr))

	print(tabulate(results))
	return results

def best_result(results):
	aics = [r[4] for r in results]
	i = np.argmin(aics)
	best_result = results[i]
	return best_result

def plot_models(data, lang, results, fmt='{}-all.png'):
	n = data.shape[0]
	x = data[:,0]
	y = data[:,1]
	est = (1-1/x)*(5-6/x)
	plt.figure(figsize=(5, 4))

	col = ''
	if len(results) == 1: col = 'r'
	lines = ["-","--","-.",":"]
	linecycler = cycle(lines)

	for i in range(len(results)):
		result = results[i]
		name, model, param, RSS, AIC, pr = result
		
		plt.plot(x, model(x, *param), col+next(linecycler),
			label='model {}, AIC = {:.2f}'.format(name, AIC))

	plt.title('Language {}'.format(lang))
	plt.plot(x, y, 'b.', label='data')
	plt.plot(x, est, 'g-', label='reference')
	plt.xlabel('$n$')
	plt.ylabel('$\\langle k^2 \\rangle$')

	plt.legend()
	plt.xscale('log')
	plt.yscale('log')
	plt.ylim((.9*np.min(y), np.max(y)*1.1))
	plt.xlim((.9*np.min(x), np.max(x)*1.1))
	if SAVE_PLOT:
		plt.savefig(FIG + fmt.format(lang), bbox_inches='tight')
		plt.close()
	else:
		plt.show()

def group_mean(data):
	df = pd.DataFrame(data, columns=('n', 'k2', 'd'))
	ndf = df.groupby(['n']).mean()
	new_data = np.array([ndf.index, ndf[METRIC]]).T
	return new_data

def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]

def table_AIC(rows, r=models_name):
	table = []
	for row in rows:
		lang, results = row
		best = best_result(results)
		best_AIC = best[4]
		row_lang = [lang]
		for result in results:
			name, model, param, RSS, AIC, pr = result
			if not name in r: continue
			row_lang.append(AIC - best_AIC)
		table.append(row_lang)
	return table

def get_table_params(rows, r=models_name):
	table = []
	for row in rows:
		lang, results = row
		row_lang = [lang]
		for result in results:
			name, model, param, RSS, AIC, pr = result
			if not name in r: continue
			for p in param:
				row_lang.append(p)
		table.append(row_lang)
	return table

def get_table_latex(table, headers):
	latex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=NUM_FMT,
		headers=headers)

	latex_table = re.sub('([+-]?[0-9]\.[0-9]*E[+-]?[0-9]*)', r'\\num{\1}',
		latex_table)
	
	return latex_table

def save_latex(s, fn):
	with open(fn, 'w') as f:
		f.write(s)

def fit_langs(langs):
	table = []
	for lang in langs:
		print('Language {}'.format(lang))
		fn = DATA.format(lang)
		data = np.genfromtxt(fn, delimiter=' ')
		data = sort_data(data)
		mean_data = group_mean(data)
		results = test_models(data, models)
		best = best_result(results)
		if PLOT_ALL:
			plot_models(data, lang, results)
			plot_models(mean_data, lang, results, fmt='{}-all-mean.png')
		if PLOT:
			plot_models(data, lang, [best], fmt='{}.png')
			plot_models(mean_data, lang, [best], fmt='{}-mean.png')
		table.append((lang, results))
	
	return table

def main():
	# First compute all the results in a big table
	all_results = fit_langs(LANGS)

	# Split models in 2 parts to avoid big tables
	models1 = models_name[:6]
	models2 = models_name[6:]

	# Create tables with AIC diff
	tableAIC1 = table_AIC(all_results, models1)
	headersAIC1 = ['Language'] + models1
	tableAIC1_tex = get_table_latex(tableAIC1, headersAIC1)
	save_latex(tableAIC1_tex, 'tableAIC1.tex')

	tableAIC2 = table_AIC(all_results, models2)
	headersAIC2 = ['Language'] + models2
	tableAIC2_tex = get_table_latex(tableAIC2, headersAIC2)
	save_latex(tableAIC2_tex, 'tableAIC2.tex')

	# Create param table
	table_param1 = get_table_params(all_results, models1)
	headers_param1 = ['Language', '1.b', '2.a', '2.b', '3.a', '3.c', '4.a', '5.a', 
	'5.b', '5.c']
	table_param2 = get_table_params(all_results, models2)
	headers_param2 = ['Language', '1+.b', '1+.d', '2+.a', '2+.b', '2+.d', '3+.a', '3+.c', 
	'3+.d', '4+.a', '4+.d', '5+.a', '5+.b', '5+.c', '5+.d']

	table_param_tex1 = get_table_latex(table_param1, headers_param1)
	table_param_tex2 = get_table_latex(table_param2, headers_param2)
	save_latex(table_param_tex1, 'table_param1.tex')
	save_latex(table_param_tex2, 'table_param2.tex')



#main()

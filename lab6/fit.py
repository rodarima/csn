import numpy as np
#import pandas as pd
from tabulate2 import tabulate
import re, json
import matplotlib.pyplot as plt
from scipy.optimize import *
from itertools import cycle
from scipy.stats.stats import pearsonr
import warnings

READ_RESULTS = False
SAVE_RESULTS = True # not READ_RESULTS
RESULTS_JSON = 'results.json'
FIG = 'fig/'
LOSS = 'soft_l1'
VERBOSE = 0
USE_EVOLUTION = False
PLOT = True #True
PLOT_ALL = True #False
SAVE_PLOT = True
METRIC = 'k2'
NUM_FMT = '.1f'
EPSILON = 5e-6 # Avoid low precision round error in the tests

N_GRAPH_MODELS = 3
GRAPH_MODELS = np.arange(N_GRAPH_MODELS) + 1

# Reproducible runs
np.random.seed(3)

DATA = 'data/model{}_hist_degree.txt'

# The first models
def m0 (t, a):			return a*t
def m1 (t, a):			return a*np.sqrt(t)
def m2 (t, a, b):		return a*t**b
def m3 (t, a, c):		return a*np.exp(c*t)
def m4 (t, a, d1):		return a * np.log(t + d1) # XXX: Assuming natural log
def m0d(t, a, d):		return a*t + d
def m1d(t, a, d):		return a*np.sqrt(t) + d
def m2d(t, a, b, d):	return a*t**b + d
def m3d(t, a, c, d):	return a*np.exp(c*t) + d
def m4d(t, a, d1, d2):	return a * np.log(t + d1) + d2


default_bounds = (-15, 14)

models = {
	'0' :{'func': m0,  'nparams': 1, 'bounds':default_bounds},
	'1' :{'func': m1,  'nparams': 1, 'bounds':default_bounds},
	'2' :{'func': m2,  'nparams': 2, 'bounds':default_bounds},
	'3' :{'func': m3,  'nparams': 2, 'bounds':default_bounds},
	'4' :{'func': m4,  'nparams': 2, 'bounds':default_bounds},

	'0+':{'func': m0d, 'nparams': 2, 'bounds':default_bounds},
	'1+':{'func': m1d, 'nparams': 2, 'bounds':default_bounds},
	'2+':{'func': m2d, 'nparams': 3, 'bounds':default_bounds},
	'3+':{'func': m3d, 'nparams': 3, 'bounds':default_bounds},
	'4+':{'func': m4d, 'nparams': 3, 'bounds':default_bounds},
}

def fit_model(data, name):
	model_info = models[name]
	bounds = model_info['bounds']
	nparams = model_info['nparams']
	model_func = model_info['func']

	bounds_vec = [bounds] * nparams

	n = data.shape[0]
	x = data[:,0]
	y = data[:,1]
	warnings.filterwarnings("ignore")

	if nparams == 0: # Reference model doesn't have params to fit
		params = []
	else:
		try:
			# Fit model params
			params, pcov = curve_fit(model_func, x, y, bounds=bounds,
				verbose=VERBOSE, loss=LOSS)
		except: # In case of non-convergence try evolution
			def diff_model(p):
				return np.sum((y-model_func(x, *p))**2)

			print('  Fit failed, running differential evolution...')
			de = differential_evolution(diff_model, bounds_vec, seed=1)
			params = de.x
	

	return list(params) # Avoid np array in order to serialize

def measure_model(data, name, params):
	model_info = models[name]
	model_func = model_info['func']
	n = data.shape[0]
	x = data[:,0]
	y = data[:,1]

	# Compute residual sum of squares RSS
	yy = model_func(x, *params)
	RSS = np.sum((y - yy)**2)

	# Compute Akaike information criterion
	p = 0
	AIC = n*np.log(2*np.pi) + n*np.log(RSS/n) + n + 2*(p + 1)

	# Compute pearson r
	pr, pval = pearsonr(y, yy)

	dict_result = {'AIC':AIC, 'RSS':RSS, 'r':pr, 'name':name,
		'params':params} 
	return dict_result

def test_models(data, models):
	results = {}
	for name in models:
		print("  Testing model {}".format(name))
		params = fit_model(data, name)
		results[name] = measure_model(data, name, params)

	#print(tabulate(results))
	return results

def best_model_name(results):
	best_AIC = float('inf')
	best_name = None
	for name, result in results.items():
		if(result['AIC'] < best_AIC):
			best_AIC = result['AIC']
			best_name = name
	return best_name

def plot_models(data, lang, results, fmt='{}-all.png'):
	n = data.shape[0]
	x = data[:,0]
	y = data[:,1]
	plt.figure(figsize=(5, 4))

	col = ''
	if len(results) == 1: col = 'r'
	lines = ["-","--","-.",":"]
	linecycler = cycle(lines)

	for name, result in results.items():
		func = models[name]['func']
		params = result['params']
		AIC = result['AIC']
		
		plt.plot(x, func(x, *params), col + next(linecycler),
			label='model {}, AIC = {:.2f}'.format(name, AIC))

	plt.title('Language {}'.format(lang))
	plt.plot(x, y, 'b.', label='data')
	plt.xlabel('$n$')
	plt.ylabel('$\\langle k^2 \\rangle$')

	#plt.legend()
	#plt.xscale('log')
	#plt.yscale('log')
	plt.ylim((.9*np.min(y), np.max(y)*1.1))
	plt.xlim((.9*np.min(x), np.max(x)*1.1))
	if SAVE_PLOT:
		plt.savefig(FIG + fmt.format(lang), bbox_inches='tight')
		plt.close()
	else:
		plt.show()

#def group_mean(data):
#	df = pd.DataFrame(data, columns=('n', 'k2', 'd'))
#	ndf = df.groupby(['n']).mean()
#	new_data = np.array([ndf.index, ndf[METRIC]]).T
#	return new_data

def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]

def table_AIC(rows, r=models):
	table = []
	for lang, results in rows.items():
		best_name = best_model_name(results)
		best_AIC = results[best_name]['AIC']
		row_lang = [lang]
		for name, result in results.items():
			if not name in r: continue
			row_lang.append(result['AIC'] - best_AIC)
		table.append(row_lang)
	return table

def get_table_params(rows, r=models):
	table = []
	for lang, results in rows.items():
		row_lang = [lang]
		for name, result in results.items():
			if not name in r: continue
			for p in result['params']:
				row_lang.append(p)
		table.append(row_lang)
	return table

def get_table_latex(table, headers, fmt=NUM_FMT):
	latex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=fmt,
		headers=headers)

	latex_table = re.sub('([+-]?[0-9]+\.[0-9]+(E[+-]?[0-9]*)?)', r'\\num{\1}',
		latex_table)
	
	return latex_table

def save_latex(s, fn):
	with open(fn, 'w') as f:
		f.write(s)

def test_equal_variance(data, model_name, params):
	pass

def fit_langs():
	table = {}
	for index_gm in GRAPH_MODELS:
		name = str(index_gm)
		print("Fitting model "+name)
		fn = DATA.format(index_gm)
		data = np.genfromtxt(fn, delimiter=' ')
		data = sort_data(data)
		data = np.log(data)
		data = data[np.isfinite(data[:,0])]
		data = data[np.isfinite(data[:,1])]
		#mean_data = group_mean(data)
		results = test_models(data, models)
		if PLOT_ALL:
			plot_models(data, name, results)
			#plot_models(mean_data, name, results, fmt='{}-all-mean.png')
		if PLOT:
			best_name = best_model_name(results)
			best = {best_name: results[best_name]}
			plot_models(data, name, best, fmt='{}.png')
			#plot_models(mean_data, name, best, fmt='{}-mean.png')
		table[name] = results
		#table.append((name, results))
	
	return table

def main():
	# First compute all the results in a big table
	if READ_RESULTS:
		with open(RESULTS_JSON, 'r') as f:
			all_results = json.load(f)
	else:
		all_results = fit_langs()

	if SAVE_RESULTS:
		with open(RESULTS_JSON, 'w') as f:
			json.dump(all_results, f)

	# Split models in 2 parts to avoid big tables
	models_name = list(models.keys())
	models1 = models_name[:6]
	models2 = models_name[6:]

	# Create tables with AIC diff
	tableAIC1 = table_AIC(all_results, models1)
	headersAIC1 = ['Language'] + models1
	tableAIC1_tex = get_table_latex(tableAIC1, headersAIC1)
	save_latex(tableAIC1_tex, 'tables/tableAIC1.tex')

	tableAIC2 = table_AIC(all_results, models2)
	headersAIC2 = ['Language'] + models2
	tableAIC2_tex = get_table_latex(tableAIC2, headersAIC2)
	save_latex(tableAIC2_tex, 'tables/tableAIC2.tex')

	# Create param table
	modelsA = models_name[:6]
	modelsB = models_name[6:9]
	modelsC = models_name[9:]

	table_param1 = get_table_params(all_results, modelsA)
	table_param2 = get_table_params(all_results, modelsB)
	table_param3 = get_table_params(all_results, modelsC)

	headers_param1 = ['Language', '1 b', '2 a', '2 b', '3 a', '3 c', '4 a',
		'5 a', '5 b', '5 c']
	headers_param2 = ['Language', '1+ b', '1+ d', '2+ a', '2+ b', '2+ d',
		'3+ a', '3+ c', '3+ d']
	headers_param3 = ['Language', '4+ a', '4+ d', '5+ a', '5+ b', '5+ c',
		'5+ d']

	fmt_param = '.3f'
	table_param_tex1 = get_table_latex(table_param1, headers_param1, fmt_param)
	table_param_tex2 = get_table_latex(table_param2, headers_param2, fmt_param)
	table_param_tex3 = get_table_latex(table_param3, headers_param3, fmt_param)

	save_latex(table_param_tex1, 'tables/table_param1.tex')
	save_latex(table_param_tex2, 'tables/table_param2.tex')
	save_latex(table_param_tex3, 'tables/table_param3.tex')



main()

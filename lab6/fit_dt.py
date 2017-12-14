import numpy as np
from scipy.special import factorial, zeta
#import pandas as pd
from tabulate2 import tabulate
import re, json
import matplotlib.pyplot as plt
from scipy.optimize import *
from itertools import cycle
from scipy.stats.stats import pearsonr
import warnings
from fitcore import *

# Reproducible runs
np.random.seed(1)
VERBOSE = 0
FIT_MEAN = True


# Model functions

class Model0(Model):
	def func(self, t, a):
		return a * t
class Model1(Model):
	def func(self, t, a):
		return a * np.sqrt(t)
class Model2(Model):
	def func(self, t, a, b):
		return a * t**b
class Model3(Model):
	def func(self, t, a, c):
		return a * np.exp(c * t)
class Model4(Model):
	params = ['a', 'd_1']
	def func(self, t, a, d1):
		return a * np.log(t + d1)

class Model0d(Model):
	name = '0+'
	def func(self, t, a, d):
		return a * t + d
class Model1d(Model):
	name = '1+'
	def func(self, t, a, d):
		return a * np.sqrt(t) + d
class Model2d(Model):
	name = '2+'
	def func(self, t, a, b, d):
		return a * t**b + d
class Model3d(Model):
	name = '3+'
	def func(self, t, a, c, d):
		return a * np.exp(c * t) + d
class Model4d(Model):
	name = '4+'
	params = ['a', 'd_1', 'd_2']
	def func(self, t, a, d1, d2):
		return a * np.log(t + d1) + d2

models = [Model0, Model1, Model2, Model3, Model4,
	Model0d, Model1d, Model2d, Model3d, Model4d]

def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]


data_dir = 'data/'
fig_dir = 'fig/'
table_dir = 'table/'
runs = 10
tracing_fn = data_dir + 'tracing_vertices.txt'
dataset_fmt = 'model{}/dt{}_r{}.txt'

N_DATASETS = 3

def aggregate_mean(data):
	df = pd.DataFrame(data, columns=('x', 'y'))
	ndf = df.groupby(['x']).mean()
	new_data = np.array([ndf.index, ndf['y']]).T
	return new_data

def fit_vertex(v):
	print(':: Fitting models for vertex {}'.format(v))
	fits = []
	for m in range(1, N_DATASETS+1):
		name = '{}'.format(m)
		data = None
		for r in range(runs):
			fn = data_dir + dataset_fmt.format(m, v, r)
			print('Reading {}'.format(fn))
			run_data = np.genfromtxt(fn, delimiter=' ')
			#data = np.log(data)
			#data = data[np.isfinite(data[:,0])]
			#data = data[np.isfinite(data[:,1])]
			#deg_sum = np.sum(data[:, 1])
			#data[:,1] /= deg_sum
			#data = sort_data(data)
			if not data is None: data = np.append(data, run_data, axis=0)
			else: data = run_data
	
		if FIT_MEAN:
			data = aggregate_mean(data)

		print('Fitting generation model {}'.format(name))
		fits.append(Fit(name, data, models, verbose=VERBOSE))

	table = TeXTable(fits)
	table1 = table.diff_measure('AIC', ' ', transpose=True)

	#print(table1)
	table.save(table1, table_dir + 'AIC_dt{}.tex'.format(v))

	# We use all_dti.png to distinguish between the best_dti.png
	fig_fmt = 'model{}/all_dt{}.png'

	for fit in fits:
		pf = PlotFit(fit)
		fig_fn = fig_dir + fig_fmt.format(fit.name, v)
		pf.comparison('AIC', 'Generation model {}. All fit models for vertex {}'.format(
			fit.name, v), fig_fn, xlabel='$t$', ylabel='$k$', mean=False)

	cmp_table = TeXTable(fits)
	tex_cmp_table = cmp_table.compare_param(transpose=True)
	fn = table_dir + 'param_dt{}.tex'.format(v)
	cmp_table.save(tex_cmp_table, fn)

def main():
	# Read the tracing vertex indices
	tracing_vertices = np.genfromtxt(tracing_fn, dtype='int')

	for v in tracing_vertices:
		fit_vertex(v)

main()

import numpy as np
from scipy.special import factorial, zeta
#import pandas as pd
from tabulate2 import tabulate
import re, json
import matplotlib.pyplot as plt
from scipy.optimize import *
from itertools import cycle
from scipy.stats.stats import pearsonr
from scipy.stats import binom, poisson
import warnings
from fitcore import *

N_GRAPH_MODELS = 3
GRAPH_MODELS = np.arange(N_GRAPH_MODELS) + 1

# Reproducible runs
np.random.seed(1)


# Model functions

class Model1(Model):
	bounds = (0.5, 10)
	params = ['\lambda']
	def __init__(self, info):
		self.info = info
	def func(self, k, l):
		return (l**k * np.exp(-l)) / (factorial(k) * (1 - np.exp(-l)))
	def mll(self, l):
		M = self.info['M']
		N = self.info['N']
		C = self.info['C']
		return -(M * np.log(l) - N*(l + np.log(1 - np.exp(-l))) - C)

class Model1c(Model):
	bounds = [[3, 6], [-5, -1], [0.1, 0.8]]
	params = ['\lambda']
	def __init__(self, info):
		self.info = info
	def func(self, k, l=4.0, dx=-3.0, s=0.4):
		return s * poisson.pmf(k + dx, l)
	def mll(self, l):
		M = self.info['M']
		N = self.info['N']
		C = self.info['C']
		return -(M * np.log(l) - N*(l + np.log(1 - np.exp(-l))) - C)

class Model1b(Model):
	bounds = [[0.01, 1.0], [2, 50], [0.5, 1], [-0.1, 0.1]]
	#params = ['\lambda', 'd']
	def __init__(self, info):
		self.info = info
	def func(self, k, p=0.5, n=10, s=1.0, dx=0):
		#N = self.info['N']
		return binom.pmf(k, n, p) * s + dx
		#return s * (l**(k+d) * np.exp(-l)) / (factorial((k+d)) * (1 - np.exp(-l)))
	def mll(self, l):
		M = self.info['M']
		N = self.info['N']
		C = self.info['C']
		return -(M * np.log(l) - N*(l + np.log(1 - np.exp(-l))) - C)

class Model2(Model):
	bounds = (0.1, 10)
	def __init__(self, info):
		self.info = info
	def func(self, k, q):
		return (1 - q)**(k - 1) * q
	def mll(self, q=0.6):
		M = self.info['M']
		N = self.info['N']
		C = self.info['C']
		return -((M - N) * np.log(1 - q) + N*np.log(q))

class Model3(Model):
	def __init__(self, info):
		self.info = info
	def func(self, k):
		zeta2 = np.pi**2/6
		return k**(-3) / zeta2

	def mll(self):
		MM = self.info['MM']
		N = self.info['N']
		return -(-2 * MM - N * np.log(np.pi ** 2 / 6))
		

class Model4(Model):
	params = ['\gamma']
	bounds = (2, 3.5)
	def __init__(self, info):
		self.info = info
	def func(self, k, g=3.0):
		return k**(-g) / zeta(g)

	def mll(self, g=3.0):
		MM = self.info['MM']
		N = self.info['N']
		return -(-g * MM - N * np.log(zeta(g)))

class Model5(Model):
	params = ['\gamma', 'k_{\max}']
	bounds = np.array([[1.8, 3.5], [+10, 30]])
	
	def __init__(self, info):
		self.info = info

	def rzeta(self, gamma, kmax):
		i = np.arange(1, kmax+1)
		return np.sum(i ** (-gamma))

	def func(self, k, g=3.0, kmax=100.0):
		return k**(-g) / self.rzeta(g, kmax)

	def mll(self, g=2.0, kmax=20.0):
		if type(g) == np.ndarray:
			kmax = g[1]
			g = g[0]

		#print((g, kmax))

		MM = self.info['MM']
		N = self.info['N']
		return -(-g * MM - N * np.log(self.rzeta(g, kmax)))


models = [Model1, Model2, Model3, Model4, Model5]
#models = [Model1c]

def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]


fig_dir = 'fig/'
table_dir = 'table/'
data_dir = 'data/'
dataset_fmt = 'model{}/dd_r{}.txt'

N_DATASETS = 3

runs = 1

def prepare_data(m):
	datasets = []
	max_nsample = 0
	for r in range(runs):
		fn = data_dir + dataset_fmt.format(m, r)
		print('Reading {}'.format(fn))
		data = np.genfromtxt(fn, delimiter=' ')

		# We will need to count degree 0 as well, assuming sorted
		max_degree = int(data[-1, 0] + 1)

		max_nsample = max(max_nsample, max_degree)
		
		data[:,1] = data[:,1] / runs
		datasets.append(data)

	all_data = np.zeros([max_nsample, 2])
	all_data[:,0] = np.arange(max_nsample)
	for data in datasets:
		all_data[np.array(data[:,0], dtype='int'), 1] += data[:,1]

	# Now we merge all datasets into data
	data = all_data[1:]
	data = data[data[:,1] >= 1]

	# Compute the info params from the data, not prob
	info = compute_info(data)

	# Compute probability
	deg_sum = np.sum(data[:, 1])
	data[:,1] /= deg_sum

	# Use log scale
	#data = np.log(data)
	return data, info

def compute_info(data):
	# We need to compute different params like N, M, C ... to init the models
	info = {}

	degree = data[:,0]
	num_nodes = data[:,1]

	# N is the number of nodes of the data
	info['N'] = np.sum(num_nodes)

	# M is the sum of the degrees
	info['M'] = np.sum(degree * num_nodes)

	# MM is the sum of degree logarithms
	info['MM'] = np.sum(num_nodes * np.log(degree))

	# C is the sum of logarithm of degree factorials
	CC = np.zeros(degree.shape)

	for i in range(degree.shape[0]):
		ki = degree[i]
		ni = num_nodes[i]
		CC[i] = ni * np.sum(np.log(np.arange(2, ki+1)))
	
	info['C'] = np.sum(CC)
	#print(info)
	return info


def fit_model(m, models):
	name = str(m)
	data, info = prepare_data(m)
	fit = Fit(name, data, models, verbose=3, info=info, mle=True)
	print('BA{}'.format(name))
	for i in range(len(models)):
		model = fit.models[i]
		print('Model{} params:'.format(i))
		print(model.fitted_params)
	return fit

def main(models):

	fits = []
	for m in range(1, N_GRAPH_MODELS+1):
		fit = fit_model(m, models)
		fits.append(fit)

		# Save plots
		fig_fmt = 'model{}/all_dd.png'

		pf = PlotFit(fit)
		fn = fig_dir + fig_fmt.format(fit.name)
		print('Saving figure {}'.format(fn))
		pf.comparison('AIC', 'Comparison AIC model {}'.format(fit.name),
			fn, xlabel='$k$', ylabel='$p(k)$', log=False)


	cmp_table = TeXTable(fits)
	tex_cmp_table = cmp_table.compare_param(transpose=True)
	fn = table_dir + 'param_dd.tex'
	cmp_table.save(tex_cmp_table, fn)


main(models)

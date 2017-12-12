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

N_GRAPH_MODELS = 3
GRAPH_MODELS = np.arange(N_GRAPH_MODELS) + 1

# Reproducible runs
np.random.seed(1)


# Model functions

class Model1(Model):
	bounds = (0, 1)
	def func(self, k, q):
		return (1 - q)**(k - 1) * q

class Model2(Model):
	bounds = (0.2, 3)
	params = ['\lambda']
	def func(self, k, l):
		return (l**k * np.exp(-l)) / (factorial(k) * (1 - np.exp(-l)))

class Model3(Model):
	def func(self, k):
		zeta2 = np.pi**2/6
		return k**(-3) / zeta2

class Model4(Model):
	params = ['\gamma']
	bounds = (2, 3.5)
	def func(self, k, g=3.0):
		return k**(-g) / zeta(g)

class Model5(Model):
	params = ['\gamma', 'k_{\max}']
	bounds = np.array([[2.5, 3.5], [+10, 130]])
	
	def rzeta(self, gamma, kmax):
		i = np.arange(1, kmax+1)
		return np.sum(i**-gamma)

	def func(self, k, g=3.0, kmax=100.0):
		return k**(-g) / self.rzeta(g, kmax)


models = [Model1, Model2, Model3, Model4, Model5]

def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]


data_dir = 'data/'
dataset_fmt = 'model{}/dd.txt'

N_DATASETS = 3

fits = []

for i in range(1, N_DATASETS+1):
	name = '{}'.format(i)
	fn = data_dir + dataset_fmt.format(i)
	data = np.genfromtxt(fn, delimiter=' ')
	#data = np.log(data)
	#data = data[np.isfinite(data[:,0])]
	#data = data[np.isfinite(data[:,1])]
	deg_sum = np.sum(data[:, 1])
	data[:,1] /= deg_sum
	data = sort_data(data)
	fits.append(Fit(name, data, models))

table = TeXTable(fits)
table1 = table.diff_measure('AIC', ' ', transpose=True)

#print(table1)
table.save(table1, 'tableAIC.tex')

fig_dir = 'fig/'
fig_fmt = 'grow_model{}_comparison.png'

for fit in fits:
	pf = PlotFit(fit)
	fn = fig_dir + fig_fmt.format(fit.name)
	pf.comparison('AIC', 'Comparison AIC model {}'.format(fit.name),
		fn, xlabel='$n$', ylabel='$k$')

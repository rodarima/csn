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
dataset_fmt = 'model{}/dt1.txt'

N_DATASETS = 3

fits = []

for i in range(1, N_DATASETS+1):
	name = '{}'.format(i)
	fn = data_dir + dataset_fmt.format(i)
	data = np.genfromtxt(fn, delimiter=' ')
	#data = np.log(data)
	#data = data[np.isfinite(data[:,0])]
	#data = data[np.isfinite(data[:,1])]
	#deg_sum = np.sum(data[:, 1])
	#data[:,1] /= deg_sum
	data = sort_data(data)
	fits.append(Fit(name, data, models))

table = TeXTable(fits)
table1 = table.diff_measure('AIC', ' ', transpose=True)

#print(table1)
table.save(table1, 'tableAIC.tex')

fig_dir = 'fig/'
fig_fmt = 'model{}_dt1_comparison.png'

for fit in fits:
	pf = PlotFit(fit)
	fn = fig_dir + fig_fmt.format(fit.name)
	pf.comparison('AIC', 'Comparison AIC model {}'.format(fit.name),
		fn, xlabel='$n$', ylabel='$k$')

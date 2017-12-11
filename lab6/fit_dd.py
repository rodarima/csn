import numpy as np
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

class Model0(Model):
	def func(self, n):
		return (1-1/n)*(5-6/n)

class Model1(Model):
	def func(self, n, b):
		return (n/2)**b
class Model2(Model):
	def func(self, n, a, b):
		return a*n**b
class Model3(Model):
	def func(self, n, a, c):
		return a*np.exp(c*n)
class Model4(Model):
	def func(self, n, a):
		return a*np.log(n)
class Model5(Model):
	def func(self, n, a, b, c):
		return a * n**b * np.exp(c*n)

class Model1d(Model):
	name = '1+'
	def func(self, n, b, d):
		return (n/2)**b + d
class Model2d(Model):
	name = '2+'
	def func(self, n, a, b, d):
		return a*n**b + d
class Model3d(Model):
	name = '3+'
	def func(self, n, a, c, d):
		return a*np.exp(c*n) + d
class Model4d(Model):
	name = '4+'
	def func(self, n, a, d):
		return a*np.log(n) + d
class Model5d(Model):
	name = '5+'
	def func(self, n, a, b, c, d):
		return a * n**b * np.exp(c*n) + d


models = [Model0, Model1, Model2, Model3, Model4, Model5,
	Model1d, Model2d, Model3d, Model4d, Model5d]


def sort_data(data):
	ind = np.argsort(data[:,0])
	return data[ind]


data_dir = 'data/'
dataset_fmt = 'model{}_hist_degree.txt'

N_DATASETS = 3

fits = []

for i in range(1, N_DATASETS+1):
	name = 'Model {}'.format(i)
	fn = data_dir + dataset_fmt.format(i)
	data = np.genfromtxt(fn, delimiter=' ')
	data = sort_data(data)
	fits.append(Fit(name, data, models))

table = TeXTable(fits)
table1 = table.diff_measure('AIC', ' ', transpose=True)

#print(table1)
table.save(table1, 'tableAIC.tex')

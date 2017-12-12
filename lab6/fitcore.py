import inspect, re
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit
from itertools import cycle
from tabulate2 import tabulate
import warnings

class Model:
	bounds = (-15, 14)
	fitted_params = None
	fitted_data = None

	def func(*args, **kargs):
		raise NotImplementedError()

	# Automatically compute param names from args
	def get_params(self):
		try:
			return self.params
		except:
			info = inspect.getfullargspec(self.func)
			all_params = info[0]
			# Skip the self and the first variable x 
			return all_params[2:]

	def get_nparams(self):
		return len(self.get_params())

	def get_bounds(self):
		# We have a global bound specification
		if type(self.bounds) == tuple:
			nparams = self.get_nparams()
			bl = [self.bounds] * nparams
		else:
			bl = self.bounds

		return bl

	def get_pair_bounds(self):
		# Get the bounds in two vectors
		bl = self.get_bounds()
		lb = [b[0] for b in bl]
		ub = [b[1] for b in bl]
		lub = [lb, ub]

		return lub

	# Automatically compute model name from the class name
	def get_name(self):
		try:
			return self.name
		except:
			name = self.__class__.__name__
			# Strip Model or Model_ from the beginning
			if name.startswith('Model_'):
				return name.replace('Model_', '')
			elif name.startswith('Model'):
				return name.replace('Model', '')
			else:
				# We return the class name as is
				return name

	def run(self, x):
		self.fitted_data = self.func(x, *self.fitted_params)
		return self.fitted_data

class Fit:
	def __init__(self, name, data, models):
		assert(len(data.shape) == 2)
		assert(data.shape[1] == 2)
		self.name = name
		self.data = data
		self.x = data[:, 0]
		self.y = data[:, 1]
		self.n = data.shape[0]
		self.models = []
		self.seed = 1
		self.verbose = 1
		self.loss = 'soft_l1'
		self.models = [m() for m in models]

		self.fit()

	def fit_model(self, model):
		if model.get_nparams() == 0:
			model.fitted_params = []
			return

		warnings.filterwarnings("ignore")

		try:
			# Fit model params
			params, pcov = curve_fit(model.func, self.x, self.y,
				bounds=model.get_pair_bounds(), verbose=self.verbose, loss=self.loss)
		except: # In case of non-convergence try evolution
			def diff_model(p):
				return np.sum((self.y - model.func(self.x, *p))**2)

			if(self.verbose):
				print('  Fit failed, running differential evolution...')
			de = differential_evolution(diff_model, model.get_bounds(), seed=self.seed)
			params = de.x

		# Avoid np array in order to call the model with *params
		model.fitted_params = list(params)

	def measure_model(self, model):
		n = self.n
		x = self.x
		y = self.y

		# Compute residual sum of squares RSS
		yy = model.run(x)
		RSS = np.sum((y - yy)**2)

		# Compute Akaike information criterion
		p = 0
		AIC = n*np.log(2*np.pi) + n*np.log(RSS/n) + n + 2*(p + 1)

		# Compute pearson r
		#pr, pval = pearsonr(y, yy)

		model.measures = {'AIC':AIC, 'RSS':RSS} 

	def best_model(self, measure):
		if measure == 'AIC':
			AICs = [m.measures[measure] for m in self.models]
			i = np.argmin(AICs)
			return self.models[i]
		else: raise NotImplementedError()

	def fit(self):
		for model in self.models:
			if(self.verbose):
				print("  Testing model {}".format(model.get_name()))
			self.fit_model(model)
			self.measure_model(model)

class TeXTable:
	def __init__(self, fits):
		self.fits = fits

	def _row_diff_measure(self, fit, measure):
		# Print all results based on the diff of a measure
		best = fit.best_model(measure)
		best_measure = best.measures[measure]
		row = []
		for model in fit.models:
			diff = model.measures[measure] - best_measure
			row.append(diff)
		return row

	def _table_diff_measure(self, measure):
		table = []
		for fit in self.fits:
			name = fit.name
			diffs = self._row_diff_measure(fit, measure)
			table.append([name] + diffs)

		return table

	def _header_diff_measure(self, title='Dataset'):
		headers = [title]
		#XXX We assume all fits have the same models
		for model in self.fits[0].models:
			name = model.get_name()
			headers.append(name)
		return headers

	def diff_measure(self, measure, title='Dataset', fmt='.3f',
		transpose=False):

		table = self._table_diff_measure(measure)
		headers = self._header_diff_measure(title)

		if transpose:
			new_headers = [title] + [row[0] for row in table]
			new_table = []
			for i in range(1, len(table[0])):
				row = [headers[i]] + [r[i] for r in table]
				new_table.append(row)

			table = new_table
			headers = new_headers

		tex_table = tabulate(table, tablefmt="latex_booktabs", floatfmt=fmt,
			headers=headers)

		tex_table = re.sub('([+-]?[0-9]+\.[0-9]+(E[+-]?[0-9]*)?)', r'\\num{\1}',
			tex_table)

		tex_table = re.sub(' inf ', r'$\\infty$', tex_table)

		return tex_table

	def save(self, s, fn):
		with open(fn, 'w') as f:
			f.write(s)


if __name__ == '__main__':
	# Test
	class Model_0(Model):
		def func(self, n, a):
			return a * n

	class Model_1(Model):
		def func(self, n, b=2):
			return (n / 2)**b

	np.random.seed(1)
	dataA = np.random.random_sample([100, 2])
	dataB = np.random.random_sample([100, 2])

	models = [Model_0, Model_1]
	fits = [Fit('A', dataA, models), Fit('B', dataB, models)]

	table = TeXTable(fits)
	table1 = table.diff_measure('AIC', title='Model', transpose=True)

	print(table1)

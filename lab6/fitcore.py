import inspect, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution, curve_fit, minimize
from itertools import cycle
from tabulate2 import tabulate
import warnings

class Model:
	bounds = (-15, 14)
	fitted_params = None
	fitted_data = None
	initial_params = 0.0

	def func(*args, **kargs):
		raise NotImplementedError()

	# Minus log likelihood
	def mll(*args, **kargs):
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

	def get_initial_params(self, mll=False):
		#print(inspect.getfullargspec(self.mll).defaults)
		# We have a global bound specification
		if mll: f = self.mll
		else: f = self.func
		default_params = inspect.getfullargspec(f).defaults
		if default_params != None and len(default_params) != 0:
			return list(default_params)
		else:
			return [self.initial_params] * self.get_nparams()

	def get_nparams(self):
		n = len(self.get_params())
		return n

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
	def __init__(self, name, data, models, verbose=0, info=None, mle=False):
		assert(len(data.shape) == 2)
		assert(data.shape[1] == 2)
		self.name = name
		self.data = data
		self.x = data[:, 0]
		self.y = data[:, 1]
		self.n = data.shape[0]
		self.models = models
		self.seed = 1
		self.verbose = verbose
		self.loss = 'soft_l1'
		self.info = info
		self.use_mle = mle

		# Models should be initialized before, so we can add fine params at the
		# beginning
		if info == None:
			self.models = [m() for m in models]
		else:
			self.models = [m(info) for m in models]
			

		self.fit()

	def fit_model_mle(self, model):
		x0 = model.get_initial_params(mll = True)
		#print(x0)
		results = minimize(model.mll, x0, method='L-BFGS-B',
			bounds=model.get_bounds(), options={'disp':self.verbose > 2})
		params = results.x
		model.fitted_params = params

	def fit_model(self, model):
		if model.get_nparams() == 0:
			model.fitted_params = []
			return

		warnings.filterwarnings("ignore")

		if self.use_mle:
			return self.fit_model_mle(model)

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

	def _header_compare_param(self, title='Dataset'):
		fit = self.fits[0]
		row = [title]
		for model in fit.models:
			model_name = model.get_name()
			for param in model.get_params():
				row.append('${} {}$'.format(model_name, param))

		return row

	def _table_compare_param(self):
		table = []
		for fit in self.fits:
			name = fit.name
			row = [name]
			for model in fit.models:
				row = row + list(model.fitted_params)
			table.append(row)

		return table

	def compare_param(self, title='Dataset', fmt='.3f',
		transpose=False):

		table = self._table_compare_param()
		headers = self._header_compare_param(title)

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

class PlotFit:

	def __init__(self, fit):
		self.fit = fit

	def plot_model(self, model, measure, style):
		# Already computed
		x = self.fit.x
		yy = model.fitted_data
		m = model.measures[measure]
		
		plt.plot(x, yy, style, label='{} {} = {:.2f}'.format(
			model.get_name(), measure, m))

	def aggregate_mean(self, data):
		df = pd.DataFrame(data, columns=('x', 'y'))
		ndf = df.groupby(['x']).mean()
		new_data = np.array([ndf.index, ndf['y']]).T
		return new_data

	def plot_data(self, mean=False):
		data = self.fit.data

		if mean: data = self.aggregate_mean(data)
		
		x = data[:,0]
		y = data[:,1]
		#print('y=')
		#print(y)

		plt.plot(x, y, 'b.', label='data')

	def scale_to_data(self):
		x = self.fit.x
		y = self.fit.y

		plt.ylim((.9 * np.min(y), np.max(y) * 1.1))
		plt.xlim((.9 * np.min(x), np.max(x) * 1.1))

	def comparison(self, measure, title, fn, xlabel='', ylabel='', mean=False, log=True):
		plt.figure(figsize=(8, 6))

		# Plot the data before the models
		self.plot_data(mean)

		lines = ["-","--","-.",":"]
		linecycler = cycle(lines)

		for model in self.fit.models:
			style = next(linecycler)
			self.plot_model(model, measure, style)

		# Plot the data after the models
		#self.plot_data(mean)

		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)

		if log:
			plt.xscale('log')
			plt.yscale('log')

		self.scale_to_data()

		plt.legend()
		plt.savefig(fn, bbox_inches='tight')
		plt.close()


if __name__ == '__main__':
	def sort_data(data):
		ind = np.argsort(data[:,0])
		return data[ind]
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

	dataA = sort_data(dataA)
	dataB = sort_data(dataB)

	models = [Model_0, Model_1]
	fits = [Fit('A', dataA, models), Fit('B', dataB, models)]

	table = TeXTable(fits)
	table1 = table.diff_measure('AIC', title='Model', transpose=True)

	print(table1)

	pf = PlotFit(fits[0])
	pf.comparison('AIC', 'AIC', '/tmp/graph.png')


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class ParallelMLPEncoding:
	def __init__(self, input_series, output_series, lag, hidden_units, lr, opt, lam, penalty, nonlinearity = 'relu'):
		# Set up networks for each output series
		self.sequentials = []
		self.p = output_series
		self.lag = lag
		self.n = input_series
		layers = list(zip([input_series * lag] + hidden_units[:-1], hidden_units))

		for target in range(output_series):
			net = torch.nn.Sequential()
			for i, (d_in, d_out) in enumerate(layers):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(hidden_units[-1], 1, bias = True))

			self.sequentials.append(net)

		# Set up optimizer
		self.loss_fn = nn.MSELoss()
		self.lr = lr
		self.lam = lam
		self.penalty = penalty

		param_list = []
		for net in self.sequentials:
			param_list = param_list + list(net.parameters())
		
		if opt == 'prox':
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
			self.train = self._train_prox

		else:
			if opt == 'adam':
				self.optimizer = optim.Adam(param_list, lr = lr)
			elif opt == 'sgd':
				self.optimizer = optim.SGD(param_list, lr = lr)
			elif opt == 'momentum':
				self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def _train_prox(self, X_batch, Y_batch):
		# Take gradient step
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate to calculate MSE
		mse_list = []
		for target in range(self.p):
			Y_var = Variable(torch.from_numpy(Y_batch[:, target][:, np.newaxis]).float())
			net = self.sequentials[target]
			y_pred = net(X_var)
			mse = self.loss_fn(y_pred, Y_var)
			mse_list.append(mse)

		# Compute total loss
		total_mse = sum(mse_list)

		# Run optimizer
		for net in self.sequentials:
			net.zero_grad()

		total_mse.backward()
		self.optimizer.step()

		# Apply prox operator
		for net in self.sequentials:
			W = list(net.parameters())[0]
			prox_operator(W, self.penalty, self.n, self.lag, self.lr, self.lam)

	def _train_builtin(self, X_batch, Y_batch):
		# Create variable for input
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate to calculate MSE
		mse_list = []
		for target in range(self.p):
			Y_var = Variable(torch.from_numpy(Y_batch[:, target][:, np.newaxis]).float())
			net = self.sequentials[target]
			y_pred = net(X_var)
			mse = self.loss_fn(y_pred, Y_var)
			mse_list.append(mse)

		# Add regularization penalties
		penalty_list = []
		for net in self.sequentials:
			W = list(net.parameters())[0]
			penalty = apply_penalty(W, self.penalty, self.n, self.lag)
			penalty_list.append(penalty)

		# Compute total loss
		total_loss = sum(mse_list) + self.lam * sum(penalty_list)

		# Run optimizer
		for net in self.sequentials:
			net.zero_grad()

		total_loss.backward()
		self.optimizer.step()

	def calculate_mse(self, X_batch, Y_batch):
		# Create variable for input
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate to calculate MSE
		mse_list = []
		for target in range(self.p):
			Y_var = Variable(torch.from_numpy(Y_batch[:, target][:, np.newaxis]).float())
			net = self.sequentials[target]
			y_pred = net(X_var)
			mse = self.loss_fn(y_pred, Y_var)
			mse_list.append(mse)

		return np.array([mse.data[0] for mse in mse_list])

	def get_weights(self):
		weights = []
		for net in self.sequentials:
			W = list(net.parameters())[0].data.numpy()
			weights.append(W)

		return weights

	def predict(self, X):
		X_var = Variable(torch.from_numpy(X).float())
		out = np.zeros((X.shape[0], self.p))
		for target in range(self.p):
			net = self.sequentials[target]
			Y_pred = net(X_var).data[:, 0].numpy()
			out[:, target] = Y_pred

		return out

def apply_penalty(W, penalty, n, lag):
	if penalty == 'group_lasso':
		loss_list = []
		for i in range(n):
			start = i * lag
			end = (i + 1) * lag
			norm = torch.norm(W[:, start:end], p = 2)
			loss_list.append(norm)
	elif penalty == 'hierarchical':
		loss_list = []
		for i in range(n):
			start = i * lag
			end = (i + 1) * lag
			for j in range(lag):
				norm = torch.norm(W[:, start:(end - j)], p = 2)
				loss_list.append(norm)
	else:
		raise ValueError('penalty must be group_lasso or hierarchical')

	return sum(loss_list)

def prox_operator(W, penalty, n, lag, lr, lam):
	if penalty == 'group_lasso':
		_prox_group_lasso(W, penalty, n, lag, lr, lam)
	elif penalty == 'hierarchical':
		_prox_hierarchical(W, penalty, n, lag, lr, lam)
	else:
		raise ValueError('penalty must be group_lasso or hierarchical')

def _prox_group_lasso(W, penalty, n, lag, lr, lam):
	'''
		Apply prox operator directly (not through prox of conjugate)
		TODO incorporate ;r
	'''
	C = W.data.clone().numpy()
	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, n), order = 'F')
	C = _prox_update(C, lam, lr)
	C = np.reshape(C, newshape = (h, l), order = 'F')
	W.data = torch.from_numpy(C)

def _prox_hierarchical(W, penalty, n, lag, lr, lam):
	''' 
		Apply prox operator for each penalty
	'''
	C = W.data.clone().numpy()
	h, l = C.shape
	C = np.reshape(C, newshape = (lag * h, n), order = 'F')
	for i in range(1, lag + 1):
		start = 0
		end = i * h
		temp = C[range(start, end), :]
		C[range(start, end), :] = _prox_update(temp, lam, lr)

	C = np.reshape(C, newshape = (h, l), order = 'F')
	W.data = torch.from_numpy(C)

def _prox_update(W, lam, lr):
	'''
		Apply prox operator to a matrix, where columns each have group lasso penalty
	'''
	norm_value = np.linalg.norm(W, axis = 0, ord = 2)
	norm_value_gt = norm_value >= lam * lr
	W[:, np.logical_not(norm_value_gt)] = 0.0
	W[:, norm_value_gt] = W[:, norm_value_gt] * (1 - lr * lam / norm_value[norm_value_gt][np.newaxis])
	return W

class ParallelMLPDecoding:
	def __init__(self, input_series, output_series, lag, series_units, fc_units, lr, opt, lam, penalty, nonlinearity = 'relu'):
		# Save important arguments
		self.p = output_series
		self.n = input_series
		self.lag = lag
		self.series_out_size = series_units[-1]

		# Set up series networks
		self.series_nets = []
		series_layers = list(zip([lag] + series_units[:-2], series_units[2:]))
		for series in range(input_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(series_layers):
				net.add_module('series fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'series relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('series sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			if len(series_units) == 1:
				d_prev = lag
			else:
				d_prev = series_units[-2]
			net.add_module('series out', nn.Linear(d_prev, series_units[-1], bias = True))
			self.series_nets.append(net)

		# Set up fully connected output networks
		self.out_nets = []
		out_layers = list(zip([series_units[-1] * input_series] + fc_units[:-1], fc_units))
		for target in range(output_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(out_layers):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(fc_units[-1], 1, bias = True))
			self.out_nets.append(net)

		# Set up optimizer
		self.loss_fn = nn.MSELoss()
		self.lr = lr
		self.lam = lam
		self.penalty = penalty

		param_list = []
		for net in (self.series_nets + self.out_nets):
			param_list = param_list + list(net.parameters())
		
		if opt == 'prox':
			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
			self.train = self._train_prox

		else:
			if opt == 'adam':
				self.optimizer = optim.Adam(param_list, lr = lr)
			elif opt == 'sgd':
				self.optimizer = optim.SGD(param_list, lr = lr)
			elif opt == 'momentum':
				self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def _train_prox(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate
		series_out = []
		for i, net in enumerate(self.series_nets):
			start = i * self.lag
			end = (i + 1) * self.lag
			series_out.append(net(X_var[:, start:end]))
		series_layer = torch.cat(series_out, dim = 1)

		mse_list = []
		for i, net in enumerate(self.out_nets):
			Y_pred = net(series_layer)
			Y_var = Variable(torch.from_numpy(Y_batch[:, i][:, np.newaxis]).float())
			mse = self.loss_fn(Y_pred, Y_var)
			mse_list.append(mse)

		# Take gradient step
		total_mse = sum(mse_list)
		for net in (self.series_nets + self.out_nets):
			net.zero_grad()

		total_mse.backward()
		self.optimizer.step()

		# Apply prox operator
		for net in self.out_nets:
			W = list(net.parameters())[0]
			prox_operator(W, self.penalty, self.n, self.series_out_size, self.lr, self.lam)

	def _train_builtin(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate
		series_out = []
		for i, net in enumerate(self.series_nets):
			start = i * self.lag
			end = (i + 1) * self.lag
			series_out.append(net(X_var[:, start:end]))
		series_layer = torch.cat(series_out, dim = 1)

		mse_list = []
		for i, net in enumerate(self.out_nets):
			Y_pred = net(series_layer)
			Y_var = Variable(torch.from_numpy(Y_batch[:, i][:, np.newaxis]).float())
			mse = self.loss_fn(Y_pred, Y_var)
			mse_list.append(mse)

		# Add penalty terms
		penalty_list = []
		for net in self.out_nets:
			W = list(net.parameters())[0]
			penalty = apply_penalty(W, self.penalty, self.n, self.series_out_size)
			penalty_list.append(penalty)

		total_loss = sum(mse_list) + self.lam * sum(penalty_list)

		# Run optimizer
		for net in (self.series_nets + self.out_nets):
			net.zero_grad()

		total_loss.backward()
		self.optimizer.step()

	def calculate_mse(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate
		series_out = []
		for i, net in enumerate(self.series_nets):
			start = i * self.lag
			end = (i + 1) * self.lag
			series_out.append(net(X_var[:, start:end]))
		series_layer = torch.cat(series_out, dim = 1)

		total_mse = []
		for i, net in enumerate(self.out_nets):
			Y_pred = net(series_layer)
			Y_var = Variable(torch.from_numpy(Y_batch[:, i][:, np.newaxis]).float())
			mse = self.loss_fn(Y_pred, Y_var)
			total_mse.append(mse)
		
		return [mse.data[0] for mse in total_mse]

	def get_weights(self):
		weights_list = []
		for net in self.out_nets:
			weights_list.append(list(net.parameters())[0].data.numpy())

		return weights_list

	def predict(self, X_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())
		out = np.zeros((X_batch.shape[0], self.p))

		# Forward propagate
		series_out = []
		for i, net in enumerate(self.series_nets):
			start = i * self.lag
			end = (i + 1) * self.lag
			series_out.append(net(X_var[:, start:end]))
		series_layer = torch.cat(series_out, dim = 1)

		for i, net in enumerate(self.out_nets):
			Y_pred = net(series_layer)
			out[:, i] = Y_pred.data[:, 0].numpy()

		return out
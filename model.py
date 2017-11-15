import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class ParallelMLP:
	def __init__(self, input_size, input_series, lag, p, hidden_units, lr, opt, lam, penalty):
		# Set up networks for each output series
		self.sequentials = []
		self.p = p
		self.lag = lag
		self.input_series = input_series
		layers = list(zip([input_size] + hidden_units[:-1], hidden_units))

		for target in range(p):
			net = torch.nn.Sequential()
			for i, (d_in, d_out) in enumerate(layers):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				net.add_module('relu%d' % i, nn.ReLU())
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

		self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)

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
		for target in range(self.p):
			net = self.sequentials[target]
			W = list(net.parameters())[0]
			prox_operator(W, self.penalty, self.input_series, self.lag, self.lr, self.lam)

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

		# Compute total MSE
		total_mse = sum(mse_list)

		# Add regularization penalties
		penalty_list = []
		for target in range(self.p):
			net = self.sequentials[target]
			W = list(net.parameters())[0]
			penalty = apply_penalty(W, self.penalty, self.input_series, self.lag)
			penalty_list.append(penalty)

		# Compute total loss
		total_loss = total_mse + self.lam * sum(penalty_list)

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
				# TODO Alex might have data organized differently, or I might be wrong
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

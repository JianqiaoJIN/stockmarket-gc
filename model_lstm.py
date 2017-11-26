import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
from itertools import chain

from regularize import *

class ParallelLSTMEncoding:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt, lam):
		# Set up networks
		self.lstms = [nn.LSTM(input_series, hidden_size, hidden_layers) for _ in range(output_series)]
		self.out_layers = [nn.Linear(hidden_size, 1) for _ in range(output_series)]

		# Save important arguments
		self.input_series = input_series
		self.output_series = output_series
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers

		# Set up optimizer
		self.loss_fn = nn.MSELoss()
		self.lam = lam
		self.lr = lr

		param_list = []
		for net in chain(self.lstms, self.out_layers):
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

	def _init_hidden(self):
		return [
			(Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
				Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))
			for _ in self.lstms
		]

	def _forward(self, X):
		n, p = X.shape
		X_var = Variable(torch.from_numpy(X).float())
		hidden = self._init_hidden()
		return [self.out_layers[target](self.lstms[target](X_var.view(n, 1, p), hidden[target])[0].view(n, self.hidden_size)) for target in range(self.output_series)]

	def _mse(self, X, Y):
		Y_pred = self._forward(X)
		Y_var = [Variable(torch.from_numpy(Y[:, target][:, np.newaxis]).float()) for target in range(self.output_series)]
		return [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.output_series)]

	def _train_prox(self, X, Y):
		mse = self._mse(X, Y)
		total_mse = sum(mse)

		# Take gradient step
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_mse.backward()
		self.optimizer.step()

		# Apply proximal operator
		[prox_operator(lstm.weight_ih_l0, 'group_lasso', self.input_series, self.lr, self.lam) for lstm in self.lstms]

	def _train_builtin(self, X, Y):
		mse = self._mse(X, Y)
		penalty = [apply_penalty(lstm.weight_ih_l0, 'group_lasso', self.input_series) for lstm in self.lstms]
		total_loss = sum(mse) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_layers)]

		total_loss.backward()
		self.optimizer.step()

	def calculate_mse(self, X, Y):
		mse = self._mse(X, Y)
		return [num.data[0] for num in mse]

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def get_weights(self):
		return [lstm.weight_ih_l0.data.numpy() for lstm in self.lstms]

class ParallelLSTMDecoding:
	def __init__(self, input_series, output_series, hidden_size, hidden_layers, fc_units, lr, opt, lam, nonlinearity = 'relu'):
		# Set up networks
		self.lstms = [nn.LSTM(1, hidden_size, hidden_layers) for _ in range(input_series)]
		
		self.out_networks = []
		for target in range(output_series):
			net = nn.Sequential()
			for i, (d_in, d_out) in enumerate(list(zip([hidden_size * input_series] + fc_units[:-1], fc_units))):
				net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
				if nonlinearity == 'relu':
					net.add_module('relu%d' % i, nn.ReLU())
				elif nonlinearity == 'sigmoid':
					net.add_module('sigmoid%d' % i, nn.Sigmoid())
				else:
					raise ValueError('nonlinearity must be "relu" or "sigmoid"')
			net.add_module('out', nn.Linear(fc_units[-1], 1, bias = True))
			self.out_networks.append(net)

		# Save important arguments
		self.input_series = input_series
		self.output_series = output_series
		self.hidden_size = hidden_size
		self.hidden_layers = hidden_layers

		# Set up optimizer
		self.loss_fn = nn.MSELoss()
		self.lam = lam
		self.lr = lr

		param_list = []
		for net in chain(self.lstms, self.out_networks):
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

	def _init_hidden(self):
		return [
			(Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
				Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))
			for _ in self.lstms
		]

	def _forward(self, X):
		n, p = X.shape
		X_var = [Variable(torch.from_numpy(X[:, in_series][:, np.newaxis]).float()) for in_series in range(self.input_series)]
		hidden = self._init_hidden()
		lstm_out = [self.lstms[in_series](X_var[in_series].view(n, 1, 1), hidden[in_series]) for in_series in range(self.input_series)]
		lstm_layer = torch.cat([out[0].view(n, self.hidden_size) for out in lstm_out], dim = 1)
		return [net(lstm_layer) for net in self.out_networks]

	def _mse(self, X, Y):
		Y_pred = self._forward(X)
		Y_var = [Variable(torch.from_numpy(Y[:, target]).float()) for target in range(self.output_series)]
		return [self.loss_fn(Y_pred[target], Y_var[target]) for target in range(self.output_series)]

	def _train_prox(self, X, Y):
		mse = self._mse(X, Y)
		total_mse = sum(mse)

		# Take gradient step
		[net.zero_grad() for net in chain(self.lstms, self.out_networks)]

		total_mse.backward()
		self.optimizer.step()

		# Apply proximal operator
		[prox_operator(list(net.parameters())[0], 'group_lasso', self.input_series, self.lr, self.lam, lag = self.hidden_size) for net in self.out_networks]

	def _train_builtin(self, X, Y):
		mse = self._mse(X, Y)
		penalty = [apply_penalty(list(net.parameters())[0], 'group_lasso', self.input_series, lag = self.hidden_size) for net in self.out_networks]
		total_loss = sum(mse) + self.lam * sum(penalty)

		# Run optimizer
		[net.zero_grad() for net in chain(self.lstms, self.out_networks)]

		total_loss.backward()
		self.optimizer.step()

	def predict(self, X):
		Y_pred = self._forward(X)
		return np.array([Y[:, 0].data.numpy() for Y in Y_pred]).T

	def calculate_mse(self, X, Y):
		mse = self._mse(X, Y)
		return [num.data[0] for num in mse]

	def get_weights(self):
		return [list(net.parameters())[0].data.numpy() for net in self.out_networks]


# class SingleLSTM:
# 	def __init__(self, input_series, output_series, hidden_size, hidden_layers, lr, opt, lam, penalty):
# 		# Set up networks
# 		self.lstm = nn.LSTM(input_series, hidden_size, hidden_layers)
# 		self.out = nn.Linear(hidden_size, output_series, bias = True)

# 		# Save important arguments
# 		self.input_series = input_series
# 		self.output_series = output_series
# 		self.hidden_size = hidden_size
# 		self.hidden_layers = hidden_layers

# 		# Set up optimizer
# 		self.loss_fn = nn.MSELoss()
# 		self.penalty = penalty
# 		self.lam = lam
# 		self.lr = lr

# 		param_list = []
# 		param_list = param_list + list(self.lstm.parameters())
# 		param_list = param_list + list(self.out.parameters())

# 		if opt == 'prox':
# 			self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
# 			self.train = self._train_prox
# 		else:
# 			if opt == 'adam':
# 				self.optimizer = optim.Adam(param_list, lr = lr)
# 			elif opt == 'sgd':
# 				self.optimizer = optim.SGD(param_list, lr = lr)
# 			elif opt == 'momentum':
# 				self.optimizer = optim.SGD(param_list, lr = lr, momentum = 0.9)
# 			else:
# 				raise ValueError('opt must be a valid option')

# 			self.train = self._train_builtin

# 	def predict(self, X, hidden = None):
# 		out = self._forward(X, hidden = hidden)
# 		return out.data.numpy()

# 	def _forward(self, X, hidden = None):
# 		if hidden is None:
# 			hidden = self.init_hidden()

# 		n, p = X.shape
# 		X_var = Variable(torch.from_numpy(X).float())

# 		lstm_out, hidden = self.lstm(X_var.view(n, 1, p), hidden)
# 		out = self.out(lstm_out.view(n, self.hidden_size))

# 		return out

# 	def init_hidden(self):
# 		return (Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)), 
# 			Variable(torch.zeros(self.hidden_layers, 1, self.hidden_size)))

# 	def _train_prox(self, X, Y, hidden = None):
# 		# Compute mse
# 		Y_pred = self._forward(X, hidden = hidden)
# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)

# 		# Take gradient step
# 		self.lstm.zero_grad()
# 		self.out.zero_grad()

# 		mse.backward()
# 		self.optimizer.step()

# 		# Apply proximal operator to first weight matrix

# 	def _train_builtin(self, X, Y, hidden = None):
# 		# Compute mse
# 		Y_pred = self._forward(X, hidden = hidden)
# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)

# 		# Compute regularization penalties on first weight matrix
# 		loss = mse

# 		# Take gradient step
# 		self.lstm.zero_grad()
# 		self.out.zero_grad()

# 		loss.backward()
# 		self.optimizer.step()

# 	def calculate_mse(self, X, Y, hidden = None):
# 		Y_pred = self._forward(X, hidden = hidden)

# 		Y_var = Variable(torch.from_numpy(Y).float())
# 		mse = self.loss_fn(Y_pred, Y_var)
# 		return mse.data[0]

# 	def get_weights(self):
# 		return self.lstm.weight_ih_l0.data.numpy()
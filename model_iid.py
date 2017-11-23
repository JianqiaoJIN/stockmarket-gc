import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np

class RegressionEncoding:
	def __init__(self, input_size, output_size, hidden_units, lr, opt, lam, penalty_groups, nonlinearity = 'relu'):
		# Set up network
		net = torch.nn.Sequential()
		layers = list(zip([input_size] + hidden_units[1:], hidden_units))

		for i, (d_in, d_out) in enumerate(layers):
			net.add_module('fc%d' % i, nn.Linear(d_in, d_out, bias = True))
			if nonlinearity == 'relu':
				net.add_module('relu%d' % i, nn.ReLU())
			elif nonlinearity == 'sigmoid':
				net.add_module('sigmoid%d' % i, nn.Sigmoid())
			else:
				raise ValueError('nonlinearity must be "relu" or "sigmoid"')
		net.add_module('out', nn.Linear(hidden_units[-1], output_size, bias = True))
		
		self.net = net

		# Set up optimizer
		self.loss_fn = nn.MSELoss()
		self.lr = lr
		self.lam = lam
		self.penalty_groups = penalty_groups

		if opt == 'prox':
			self.optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
			self.train = self._train_prox

		else:
			if opt == 'adam':
				self.optimizer = optim.Adam(net.parameters(), lr = lr)
			elif opt == 'sgd':
				self.optimizer = optim.SGD(net.parameters(), lr = lr)
			elif opt == 'momentum':
				self.optimizer = optim.SGD(net.parameters(), lr = lr, momentum = 0.9)
			else:
				raise ValueError('opt must be a valid option')

			self.train = self._train_builtin

	def _train_builtin(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())
		Y_var = Variable(torch.from_numpy(Y_batch).float())

		# Forward propagate to calculate MSE
		Y_pred = self.net(X_var)
		mse = self.loss_fn(Y_pred, Y_var)

		# Add regularization penalties
		W = list(self.net.parameters())[0]
		penalty_list = []
		for start, end in zip(self.penalty_groups[:-1], self.penalty_groups[1:]):
			penalty_list.append(torch.norm(W[:, start:end], p = 2))

		# Compute total loss
		total_loss = mse + self.lam * sum(penalty_list)

		# Run optimizer
		self.net.zero_grad()

		total_loss.backward()
		self.optimizer.step()

	def _train_prox(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())
		Y_var = Variable(torch.from_numpy(Y_batch).float())

		# Forward propagate to calculate MSE
		Y_pred = self.net(X_var)
		mse = self.loss_fn(Y_pred, Y_var)

		# Run optimizer
		self.net.zero_grad()
		mse.backward()
		self.optimizer.step()

		# Apply prox operator
		W = list(self.net.parameters())[0]
		C = W.data.clone().numpy()
		for start, end in zip(self.penalty_groups[:-1], self.penalty_groups[1:]):
			norm = np.linalg.norm(C[:, start:end], ord = 2)
			if norm >= self.lr * self.lam:
				C[:, start:end] = C[:, start:end] * (1 - self.lr * self.lam / norm)
			else: 
				C[:, start:end] = 0.0
		W.data = torch.from_numpy(C)

	def calculate_mse(self, X_batch, Y_batch):
		# Create variables
		X_var = Variable(torch.from_numpy(X_batch).float())
		Y_var = Variable(torch.from_numpy(Y_batch).float())

		# Forward propagate
		Y_pred = self.net(X_var)

		# Calculate MSE
		mse = self.loss_fn(Y_pred, Y_var)
		return mse.data.numpy()[0]

	def get_weights(self):
		return list(self.net.parameters())[0].data.numpy()

	def predict(self, X_batch):
		# Create variable
		X_var = Variable(torch.from_numpy(X_batch).float())

		# Forward propagate
		Y_pred = self.net(X_var)
		return Y_pred.data.numpy()

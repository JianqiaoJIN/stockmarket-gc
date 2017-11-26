import numpy as np
import pandas as pd
import argparse
import os
from itertools import product
import pickle
import shutil
import sys
import matplotlib.pyplot as plt

from data_processing import *
from experiment import run_experiment
from model_lstm import *

# Parse command line arguments
mbsize = None
lam_list = [0.1, 2.0, 10.0]
lam = 0.1
opt_type = 'adam'
seed = 12345
nepoch = 1000
arch = 1
lr = 0.005
filename = 'Data/var.csv'

# Prepare data
data = pd.read_csv(filename, dtype = float, header = 0, sep = ',')
data_values = data.values[:, 1:]

X_train, X_val = split_data(data_values, validation = 0.1)
Y_train = X_train[1:, :]
X_train = X_train[:-1, :]
Y_val = X_val[1:, :]
X_val = X_val[:-1, :]

p_in = X_val.shape[1]
p_out = Y_val.shape[1]

# Minibatching options
if mbsize is None:
	batchsize = 0
else:
	batchsize = mbsize

# Determine architecture
if arch == 1:
	hidden_size = 1
	hidden_layers = 1
	fc_units = [5]
elif arch == 2:
	hidden_size = 2
	hidden_layers = 1
	fc_units = [5]
elif arch == 3:
	hidden_size = 1
	hidden_layers = 2
	fc_units = [10] 
elif arch == 4:
	hidden_size = 2
	hidden_layers = 2
	fc_units = [10]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []

# Run optimizations
for lam in lam_list:

	# Get model
	model = ParallelLSTMDecoding(p_in, p_out, hidden_size, hidden_layers, fc_units, lr, opt_type, lam)

	# Run experiment
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_experiment(model, X_train, Y_train, X_val, Y_val, 
		nepoch, mbsize = mbsize, predictions = True, loss_check = 1)
	
	# Create GC estimate grid
	GC_est = np.zeros((p_out, p_in))
	for target in range(p_out):
		W = weights_list[target]
		for candidate in range(p_in):
			start = candidate * lag
			end = (candidate + 1) * lag
			GC_est[target, candidate] = np.linalg.norm(W[:, range(start, end)], ord = 2)
	
	# Save results
	results_dict = {
		'train_loss': train_loss,
		'val_loss': val_loss,
		'weights_list': weights_list,
		'GC_est': GC_est,
		'forecasts_train': forecasts_train,
		'forecasts_val': forecasts_val
	}
	results.append(results_dict)

# Save results
with open('var_experiment.out', 'wb') as f:
	pickle.dump(results, f)

for lam, results_dict in zip(lam_list, results):
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.title('lam = %f' % lam)
	plt.show()

for lam, results_dict in zip(lam_list, results):
	Y = Y_val
	fc = results_dict['forecasts_val']
	fig = plt.figure(figsize = (10, 6))
	for i in range(p_out):
		ax = fig.add_subplot(2, 5, i + 1)
		ax.plot(Y[:, i])
		ax.plot(fc[:, i])
	plt.suptitle('lam = %f' % lam)
	plt.show()
	Y = Y_train
	fc = results_dict['forecasts_train']
	fig = plt.figure(figsize = (10, 6))
	for i in range(p_out):
		ax = fig.add_subplot(2, 5, i + 1)
		ax.plot(Y[:, i])
		ax.plot(fc[:, i])
	plt.suptitle('lam = %f' % lam)
	plt.show()


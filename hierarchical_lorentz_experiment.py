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
from mlp_experiment import run_encoding_experiment

# Parse command line arguments
lag = 3
mbsize = None
lam_list = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
penalty_type = 'hierarchical'
opt_type = 'prox'
seed = 12345
nepoch = 10000
arch = 1
lr = 0.005
filename = 'Data/lorentz.csv'

# Prepare data
data = pd.read_csv(filename, dtype = float, header = 0, sep = ',')
data_values = data.values[:, 1:]

X_train, Y_train, X_val, Y_val = format_ts_data(data_values, lag = lag, validation = 0.1)

T, p_out = Y_train.shape
p_in = int(X_train.shape[1] / lag)

# Create filenames
if mbsize is None:
	batchsize = 0
else:
	batchsize = mbsize

# Determine architecture
if arch == 1:
	hidden_units = [p_in]
elif arch == 2:
	hidden_units = [p_in, p_in]
elif arch == 3:
	hidden_units = [2 * p_in]
elif arch == 4:
	hidden_units = [2 * p_in, 2 * p_in]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []

# Run optimizations
for lam in lam_list:

	# Run experiment
	np.random.seed(seed)
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_encoding_experiment(X_train, Y_train, X_val, Y_val, 
		lag, nepoch, lr, lam, penalty_type, hidden_units, opt_type, mbsize = mbsize, predictions = True)
	
	# Create GC estimate grid
	GC_est = np.zeros((p_out, p_in))
	for target in range(p_out):
		W = weights_list[target]
		for candidate in range(p_in):
			start = candidate * lag
			end = (candidate + 1) * lag
			GC_est[target, candidate] = np.linalg.norm(W[:, range(start, end)], ord = 2)

	# Create lagged GC grid
	lagged_GC_est = np.zeros((p_out, lag, p_in))
	for target in range(p_out):
		W = weights_list[target]
		h, l = W.shape
		C = np.reshape(W, newshape = (lag * h, p_in), order = 'F')
		for i in range(1, lag + 1):
			start = 0
			end = h * i
			lagged_GC_est[target, i - 1, :] = np.linalg.norm(C[range(start, end), :], axis = 0, ord = 2)
	
	# Save results
	results_dict = {
		'train_loss': train_loss,
		'val_loss': val_loss,
		'weights_list': weights_list,
		'GC_est': GC_est,
		'lagged_GC': lagged_GC_est,
		'forecasts_train': forecasts_train,
		'forecasts_val': forecasts_val
	}
	results.append(results_dict)

# Save results
with open('lorentz_experiment.out', 'wb') as f:
	pickle.dump(results, f)

# Loss plots
# for results_dict in results:
# 	fig = plt.figure(figsize = (8, 5))
# 	ax = fig.add_subplot(1, 1, 1)
# 	ax.plot(results_dict['train_loss'], color = 'orange')
# 	ax.plot(results_dict['val_loss'], color = 'blue')
# 	plt.show()

# GC recovery plots
for results_dict in results:
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.show()

# Hierarchical GC plots
for results_dict in results:
	lagged_GC = results_dict['lagged_GC']
	for target in range(p_out):
		plt.imshow(lagged_GC[target, :, :], cmap = 'gray')
		plt.show()

# GC recovery plots for 8 different nepochs
# fig = plt.figure()
# for i, (result, n) in enumerate(zip(results, nepoch_list), 1):
# 	ax = fig.add_subplot(2, 4, i)
# 	ax.imshow(result['GC_est'], cmap = 'gray')
# 	ax.xaxis.set_visible(False)
# 	ax.yaxis.set_visible(False)
# 	ax.set_title('nepochs = %d' % n)
# plt.suptitle('Lorentz experiment, lam = 0.2, hidden = [10], lr = 0.0001')
# plt.show()


# for results_dict in results:
# 	Y = Y_val
# 	fc = results_dict['forecasts_val']
# 	fig = plt.figure(figsize = (10, 6))
# 	for i in range(p_out):
# 		ax = fig.add_subplot(2, 5, i + 1)
# 		ax.plot(Y[:, i])
# 		ax.plot(fc[:, i])
# 	plt.show()
# 	Y = Y_train
# 	fc = results_dict['forecasts_train']
# 	fig = plt.figure(figsize = (10, 6))
# 	for i in range(p_out):
# 		ax = fig.add_subplot(2, 5, i + 1)
# 		ax.plot(Y[:, i])
# 		ax.plot(fc[:, i])
# 	plt.show()
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
from mlp_experiment import run_decoding_experiment

# Parse command line arguments
lag = 1
mbsize = None
lam_list = [5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01]
lam_list = [10.0, 8.0, 6.0, 4.0, 2.0]
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nepoch = 10000
arch = 1
lr = 0.005
filename = 'Data/var.csv'

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
	series_units = [1]
	fc_units = [p_in]
elif arch == 2:
	series_units = [2]
	fc_units = [2 * p_in]
elif arch == 3:
	series_units = [2]
	fc_units = [p_in, p_in]
elif arch == 4:
	series_units = [2, 2]
	fc_units = [2 * p_in, 2 * p_in]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []

# Run optimizations
for lam in lam_list:

	# Run experiment
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_decoding_experiment(X_train, Y_train, X_val, Y_val, 
		lag, nepoch, lr, lam, penalty_type, series_units, fc_units, opt_type, mbsize = mbsize, predictions = True)
	
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

# Validation loss plots

fig, ax_list = plt.subplots(1, 3, sharey = 'row')

for results_dict, ax in zip(results, ax_list):
	ax.plot(results_dict['val_loss'])
plt.show()

# GC recovery plots

for results_dict in results:
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.show()

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
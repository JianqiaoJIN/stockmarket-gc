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
from model_mlp import *

# Parse command line arguments
lag = 3
mbsize = None
lam_list = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.1]
lam_list = [0.3]
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nepoch = 1000
arch = 4
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
	series_units = [1]
	fc_units = [p_in]
elif arch == 2:
	series_units = [2]
	fc_units = [2 * p_in]
elif arch == 3:
	series_units = [2]
	fc_units = [p_in, p_in]
elif arch == 4:
	series_units = [3]
	fc_units = [2 * p_in, 2 * p_in]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []

# Run optimizations
for lam in lam_list:

	# Get model
	model = ParallelMLPDecoding(p_in, p_out, lag, series_units, fc_units, lr, opt_type, lam, penalty_type)

	# Run experiment
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_experiment(model, X_train, Y_train, X_val, Y_val, 
		nepoch, mbsize = mbsize, predictions = True)
	
	# Create GC estimate grid
	GC_est = np.zeros((p_out, p_in))
	for target in range(p_out):
		W = weights_list[target]
		for candidate in range(p_in):
			start = candidate * series_units[-1]
			end = (candidate + 1) * series_units[-1]
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
with open('lorentz_experiment.out', 'wb') as f:
	pickle.dump(results, f)

# Loss plots
for la, results_dict in zip(lam_list, results):
	fig = plt.figure(figsize = (8, 5))
	ax = fig.add_subplot(1, 1, 1)
	ax.plot(results_dict['train_loss'], color = 'orange')
	ax.plot(results_dict['val_loss'], color = 'blue')
	ax.set_title('lam = %f' % lam)
	plt.show()

# GC recovery plots
for lam, results_dict in zip(lam_list, results):
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.title('lam = %f' % lam)
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

for results_dict in results:
	Y = Y_val
	fc = results_dict['forecasts_val']
	fig = plt.figure(figsize = (10, 6))
	for i in range(p_out):
		ax = fig.add_subplot(2, 5, i + 1)
		ax.plot(Y[:, i])
		ax.plot(fc[:, i])
	plt.show()
	Y = Y_train
	fc = results_dict['forecasts_train']
	fig = plt.figure(figsize = (10, 6))
	for i in range(p_out):
		ax = fig.add_subplot(2, 5, i + 1)
		ax.plot(Y[:, i])
		ax.plot(fc[:, i])
	plt.show()
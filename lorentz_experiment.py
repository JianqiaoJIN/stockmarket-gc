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
from mlp_experiment import run_experiment

# Parse command line arguments
lag = 3
mbsize = None
lam_list = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0]
lam_list = [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nepoch = 10000
arch = 1
lr = 0.001
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
	train_loss, val_loss, weights_list = run_experiment(X_train, Y_train, X_val, Y_val, 
		lag, nepoch, lr, lam, penalty_type, hidden_units, opt_type, mbsize = mbsize)
	
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
		'GC_est': GC_est
	}
	results.append(results_dict)

# Save results
with open('lorentz_results.out', 'wb') as f:
	pickle.dump(results, f)

# Validation loss plots

# fig, ax_list = plt.subplots(1, 3, sharey = 'row')

# for results_dict, ax in zip(results, ax_list):
# 	ax.plot(results_dict['val_loss'])
# plt.show()

# GC recovery plots

for results_dict in results:
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.show()
	
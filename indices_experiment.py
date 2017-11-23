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
lag = 5
mbsize = None
lam_list = [0.01, 0.05, 0.1, 0.5]
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nepoch = 10000
arch = 2
lr = 0.001
filename = 'Data/international_indices_returns.csv'

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
	train_loss, val_loss, weights_list = run_encoding_experiment(X_train, Y_train, X_val, Y_val, 
		lag, nepoch, lr, lam, penalty_type, hidden_units, opt_type, mbsize = mbsize, nonlinearity = 'sigmoid')
	
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
with open('international_indices.out', 'wb') as f:
	pickle.dump(results, f)

for results_dict in results:
	plt.imshow(results_dict['GC_est'], cmap = 'gray')
	plt.show()
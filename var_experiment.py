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
lag = 1
mbsize = None
lam_list = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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

	# Get model
	model = ParallelMLPEncoding(p_in, p_out, lag, hidden_units, lr, opt_type, lam, penalty_type)
	
	# Run experiment
	train_loss, val_loss, weights_list = run_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = mbsize)
	
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


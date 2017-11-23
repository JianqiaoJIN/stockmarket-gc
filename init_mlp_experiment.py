import numpy as np
import pandas as pd
import argparse
import os
from itertools import product
import pickle
import shutil
import sys

from data_processing import *
from mlp_experiment import run_encoding_experiment

# Parse command line arguments
lag = 1
mbsize = None
lam = 1.2
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nrestarts = 1
nepoch = 10000
arch = 1
lr_list = [0.001]
outdir = 'out/MLP Experiment'
checkdir = 'checkpoints/MLP Experiment'
filename = 'Data/var.csv'

# Modify outdir and checkdir
dname = filename.split('.')[0].split('/')[-1]
outdir = os.path.join(outdir, dname)
checkdir = os.path.join(checkdir, dname)

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

experiment_name = 'mlp_experiment'
experiment_name += '_lag-%d_lam-%e'
experiment_name += '_pentype-%s_opt-%s_nepoch-%d'
experiment_name += '_mbsize-%d_seed-%d_nrestarts-%d'
experiment_name = experiment_name % (lag, lam, penalty_type, opt_type, nepoch, batchsize, seed, nrestarts)

outname = os.path.join(outdir, experiment_name + '.data')

# Ensure that experiment has not already been performed
if os.path.isfile(outname):
	print('Experiment has already been performed')
	sys.exit()

# Create out dir, if it doesn't exist
if not os.path.exists(outdir):
	os.makedirs(outdir)

# Create checkpoint dir, if it doesn't exist
if not os.path.exists(os.path.join(checkdir, experiment_name)):
	os.makedirs(os.path.join(checkdir, experiment_name))

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

# Prepare containers for best results
best_lr = [None] * p_out
best_train_loss = [None] * p_out
best_val_loss = [np.inf] * p_out
best_weights = [None] * p_out

# Run optimizations
for lr, restart in product(lr_list, range(nrestarts)):

	# Create checkpoint filename
	checkname = os.path.join(checkdir, experiment_name, 'checkpoint-%e-%d.data' % (lr, restart))

	# See if checkpoint already exists
	if not os.path.isfile(checkname):
		print('Running experiment for lr=%e, restart=%d' %(lr, restart))

		# Run experiment
		train_loss, val_loss, weights_list = run_encoding_experiment(X_train, Y_train, X_val, Y_val, 
			lag, nepoch, lr, lam, penalty_type, hidden_units, opt_type, mbsize = mbsize)

		# Save results
		results = {
			'train_loss': train_loss,
			'val_loss': val_loss,
			'weights_list': weights_list
		}

		with open(checkname, 'wb') as f:
			pickle.dump(results, f)

	else:
		print('Loading results for lr=%e, restart=%d' % (lr, restart))

		with open(checkname, 'rb') as f:
			results = pickle.load(f)

		train_loss = results['train_loss']
		val_loss = results['val_loss']
		weights_list = results['weights_list']

	# Check if these are best results yet
	for target in range(p_out):
		if val_loss[-1, target] < best_val_loss[target]:
			best_val_loss[target] = val_loss[-1, target]
			best_train_loss[target] = train_loss[-1, target]
			best_weights[target] = weights_list[target]
			best_lr[target] = lr

# Create GC estimate grid
GC_est = np.zeros((p_out, p_in))
for target in range(p_out):
	W = best_weights[target]
	for candidate in range(p_in):
		start = candidate * lag
		end = (candidate + 1) * lag
		GC_est[target, candidate] = np.linalg.norm(W[:, range(start, end)], ord = 2)

# Format results
experiment_params = {
	'datafile': filename,
	'columns': data.columns.values[1:],
	'lag': lag
}

training_params = {
	'lam': lam,
	'lr': lr_list,
	'nrestarts' : nrestarts,
	'penalty_type': penalty_type,
	'opt_type': opt_type,
	'nepoch': nepoch,
	'mbsize': batchsize
}

out_dict = {
	'best_lr': best_lr,
	'best_train_loss': best_train_loss,
	'best_val_loss': best_val_loss,
	'weights': weights_list,
	'GC_est': GC_est,
	'experiment_params': experiment_params,
	'training_params': training_params
}

# Save results
#with open(outname, 'wb') as f:
#	pickle.dump(out_dict, f)

# Delete checkpoint files
shutil.rmtree(os.path.join(checkdir, experiment_name))


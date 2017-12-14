import numpy as np
import pandas as pd
import argparse
import os
from itertools import product
import pickle
import shutil
import sys

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase

from data_processing import *
from experiment import run_experiment
from model_mlp import *

# Parse command line arguments
lag = 6
mbsize = None
#lam_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.0001]
lam_list = [0.5, 0.1, 0.01]
seed_list = [987, 876, 765, 654, 543, 432, 321, 210, 109, 98]
penalty_type = 'group_lasso'
opt_type = 'prox'
seed = 12345
nepoch = 5000
arch = 1
lr = 0.001
filename = 'Data/Libor CPI/log_returns.csv'

# Prepare data
data = np.loadtxt(filename, delimiter = ',')
data = normalize(data)

X_train, Y_train, X_val, Y_val = format_ts_data(data, lag = lag, validation = 0.1)

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

hidden_units = [2]

# List for results
results = []

# Run optimizations
for lam, seed in product(lam_list, seed_list):

	# Get model
	torch.manual_seed(seed)
	model = ParallelMLPEncoding(p_in, p_out, lag, hidden_units, lr, opt_type, lam, penalty_type)
	
	# Run experiment
	train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = mbsize, predictions = True, loss_check = 50, verbose = False)
	
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
		'lam': lam,
		'seed': seed,
		'train_loss': train_loss,
		'val_loss': val_loss,
		'weights_list': weights_list,
		'GC_est': GC_est,
		'forecasts_train': forecasts_train,
		'forecasts_val': forecasts_val
	}
	results.append(results_dict)

# Save results
with open('cpi_lib.out', 'wb') as f:
	pickle.dump(results, f)

# for lam, results_dict in zip(lam_list, results):
# 	plt.imshow(results_dict['GC_est'], cmap = 'gray')
# 	plt.title('lam = %f' % lam)
# 	plt.show()

# for lam, results_dict in zip(lam_list, results):
# 	Y = Y_val
# 	fc = results_dict['forecasts_val']
# 	fig = plt.figure(figsize = (10, 6))
# 	for i in range(p_out):
# 		ax = fig.add_subplot(2, 5, i + 1)
# 		ax.plot(Y[:, i])
# 		ax.plot(fc[:, i])
# 	plt.suptitle('lam = %f' % lam)
# 	plt.show()
# 	Y = Y_train
# 	fc = results_dict['forecasts_train']
# 	fig = plt.figure(figsize = (10, 6))
# 	for i in range(p_out):
# 		ax = fig.add_subplot(2, 5, i + 1)
# 		ax.plot(Y[:, i])
# 		ax.plot(fc[:, i])
# 	plt.suptitle('lam = %f' % lam)
# 	plt.show()

metric_list = {
	0.5: {
		'GC': [],
		'train_loss': [],
		'val_loss': []
	},
	0.1: {
		'GC': [],
		'train_loss': [],
		'val_loss': []
	},
	0.01: {
		'GC': [],
		'train_loss': [],
		'val_loss': []
	}
}

for result, (lam, seed) in zip(results, product(lam_list, seed_list)):
	GC = np.reshape(result['GC_est'], (4, ))
	metric_list[lam]['GC'].append(GC)
	metric_list[lam]['train_loss'].append(result['train_loss'])
	metric_list[lam]['val_loss'].append(result['val_loss'])


class AnyObjectHandler(HandlerBase):
	def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
		l = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=orig_handle[0])
		return [l]

fig, axarr = plt.subplots(3, 3, sharey = 'row', figsize = (12, 8))
for i, lam in enumerate(lam_list):
	metric = metric_list[lam]
	
	GC = metric['GC']
	GC = np.array(GC)

	train_loss = metric['train_loss']
	val_loss = metric['val_loss']

	train_loss1 = [l[:, 0] for l in train_loss]
	train_loss2 = [l[:, 1] for l in train_loss]
	val_loss1 = [l[:, 0] for l in val_loss]
	val_loss2 = [l[:, 1] for l in val_loss]

	# Plot GC boxplots
	ax = axarr[0, i]
	ax.boxplot(GC)
	ax.set_xticklabels(['CPI->CPI', 'LIB->CPI', 'CPI->LIB', 'LIB->LIB'], fontsize = 10, rotation = 20)

	# Plot loss 1
	ax = axarr[1, i]
	ax.plot(np.arange(0, 5000, 50), np.array(train_loss1).T, color = 'blue', alpha = 0.3)
	ax.plot(np.arange(0, 5000, 50), np.array(val_loss1).T, color = 'orange', alpha = 0.3)
	ax.legend([('blue', '--'), ('orange', '--')], ['Train', 'Validation'], handler_map={tuple: AnyObjectHandler()}, loc = 'upper right', fontsize = 8)
	ax.set_xlabel('Epoch', fontsize = 10)

	# Plot loss 2
	ax = axarr[2, i]
	ax.plot(np.arange(0, 5000, 50), np.array(train_loss2).T, color = 'blue', alpha = 0.3)
	ax.plot(np.arange(0, 5000, 50), np.array(val_loss2).T, color = 'orange', alpha = 0.3)
	ax.legend([('blue', '--'), ('orange', '--')], ['Train', 'Validation'], handler_map={tuple: AnyObjectHandler()}, loc = 'upper right', fontsize = 8)
	ax.set_xlabel('Epoch', fontsize = 10)

for i, lam in enumerate(lam_list):
	ax = axarr[0, i]
	ax.set_title(r'$\lambda = %.2f$' % lam, fontsize = 14)

for i, label in enumerate(['GC Metric', 'CPI MSE', 'Libor MSE']):
	ax = axarr[i, 0]
	ax.set_ylabel(label, fontsize = 14)

plt.tight_layout()
plt.subplots_adjust(hspace = 0.4, bottom = 0.07)
plt.show()

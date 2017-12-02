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
from model_iid import *

from mnist import MNIST

def load_dataset():
	mndata = MNIST('/Users/icovert//python-mnist/data/')
	X_train, labels_train = map(np.array, mndata.load_training())
	X_test, labels_test = map(np.array, mndata.load_testing())
	X_train = X_train / 255.0
	X_test = X_test / 255.0
	return X_train, labels_train, X_test, labels_test

# Parse command line arguments
mbsize = 32
lam_list = [1.0, 0.1, 0.01, 0.001, 0.0001]
opt = 'prox'
seed = 123
nepoch = 10
arch = 1
lr = 0.001

X_train, Y_train, X_val, Y_val = load_dataset()

N, p_in = X_train.shape
p_out = 10

penalty_groups = list(np.arange(0, p_in + 1))

# Determine architecture
if arch == 1:
	hidden_units = [100]
elif arch == 2:
	hidden_units = [100, 100]
elif arch == 3:
	hidden_units = [500, 500]
elif arch == 4:
	hidden_units = [500, 500, 250]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []

# Run optimizations
for lam in lam_list:
	filename = 'mnist_lam-%e.out' % lam

	if os.path.isfile(filename):
		print('Skipping experiment with lam = %e' % lam)

		with open(filename, 'rb') as f:
			results_dict = pickle.load(f)

	else:
		print('Training network with lam = %f' % lam)

		# Get model
		model = IIDEncoding(p_in, p_out, hidden_units, lr, opt, lam, penalty_groups, nonlinearity = 'sigmoid', task = 'classification')

		# Run experiment
		train_loss, val_loss, train_accuracy, val_accuracy, weights = run_experiment(model, X_train, Y_train, X_val, Y_val, 
			nepoch, mbsize = mbsize, loss_check = 1)
		
		# Perform variable selection
		GC_est = np.zeros(len(penalty_groups) - 1)
		adj_GC_est = np.zeros(len(penalty_groups) - 1)
		for i, (start, end) in enumerate(list(zip(penalty_groups[:-1], penalty_groups[1:]))):
			GC_est[i] = np.linalg.norm(weights[:, start:end], ord = 2)
			adj_GC_est[i] = np.linalg.norm(weights[:, start:end], ord = 2) / (end - start)
			
		# Save results
		results_dict = {
			'train_loss': train_loss,
			'val_loss': val_loss,
			'train_accuracy': train_accuracy,
			'val_accuracy': val_accuracy,
			'weights': weights,
			'GC_est': GC_est,
			'adj_GC_est': adj_GC_est
		}

		with open(filename, 'wb') as f:
			pickle.dump(results_dict, f)
	
	results.append(results_dict)

with open('mnist.out', 'wb') as f:
	pickle.dump(results, f)

# for lam, result in zip(lam_list, results):
# 	GC = result['GC_est']
# 	plt.imshow(np.reshape(GC, (28, 28)), cmap = 'gray')
# 	plt.title(r'$\lambda = %.4f$' % lam)
# 	plt.show()

# for lam, result in zip(lam_list, results):
# 	plt.plot(result['train_loss'])
# 	plt.title(r'$\lambda = %.4f$' % lam)
# 	plt.show()

cmap = plt.cm.gray
cmap.set_bad(color = 'cyan')
fig, axarr = plt.subplots(2, 4, sharey = 'row', figsize = (10, 5))
for i, (lam, result) in enumerate(list(zip(lam_list[:-1], results[:-1]))):
	GC = result['GC_est']
	train_accuracy = result['train_accuracy']
	val_accuracy = result['val_accuracy']

	# Plot GC image
	ax = axarr[0, i]
	masked = np.ma.masked_where(GC == 0, GC)
	ax.imshow(np.reshape(masked, (28, 28)), cmap = cmap)
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().set_ticks([])

	# Plot accuracy
	ax = axarr[1, i]
	ax.plot(train_accuracy, label = 'Train')
	ax.plot(val_accuracy, label = 'Validation')
	ax.legend(loc = 'center right', fontsize = 10)
	ax.set_xlabel('Epoch')

for i, label in enumerate(['GC Metric', 'Accuracy']):
	ax = axarr[i, 0]
	ax.set_ylabel(label)

for i, lam in enumerate(lam_list[:-1]):
	ax = axarr[0, i]
	ax.set_title(r'$\lambda = %.4f$' % lam)

plt.subplots_adjust(left = 0.1)
plt.tight_layout()
plt.show()
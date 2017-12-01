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

	# Get model
	model = IIDEncoding(p_in, p_out, hidden_units, lr, opt, lam, penalty_groups, nonlinearity = 'sigmoid', task = 'classification')

	# Run experiment
	train_loss, val_loss, weights = run_experiment(model, X_train, Y_train, X_val, Y_val, 
		nepoch, mbsize = mbsize)
	
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
		'weights': weights,
		'GC_est': GC_est,
		'adj_GC_est': adj_GC_est
	}
	results.append(results_dict)

with open('mnist.out', 'wb') as f:
	pickle.dump(results, f)

for lam, result in zip(lam_list, results):
	GC = result['GC_est']
	plt.imshow(np.reshape(GC, (28, 28)), cmap = 'gray')
	plt.title('lam = %f' % lam)
	# plt.show()
	plt.savefig('mnist_lam_%f.png' % lam)
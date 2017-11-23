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
from regression_experiment import run_iid_experiment

# Parse command line arguments
mbsize = None
lam_list = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001, 0.00000001]
opt = 'prox'
seed = 123
nepoch = 30000
arch = 4
lr = 0.001
filename = 'Data/day_processed.csv'

# Prepare data
data = pd.read_csv(filename, header = 0, sep = ',')
data_values = data.values[:, 2:].astype(float)

# Randomize order
np.random.seed(987)
np.random.shuffle(data_values)

val_portion = 0.1
X = data_values[:, :-3]
Y = data_values[:, -1]
Y = Y / np.max(Y)
N = X.shape[0]
N_val = int(N * val_portion)
N_train = N - N_val
X_train = X[range(N_train), :]
X_val = X[range(N_train, N), :]
Y_train = Y[range(N_train)]
Y_val = Y[range(N_train, N)]

# Encode groups for penalties
penalty_groups = [0, 4, 5, 17, 18, 25, 26, 30, 31, 32, 33, 34]
input_groups = ['season', 'year', 'month', 'holiday', 'weekday', 'workingday', 'weather', 'temp', 'atemp', 'hum', 'windspeed']

# Create filenames
if mbsize is None:
	batchsize = 0
else:
	batchsize = mbsize

# Determine architecture
if arch == 1:
	hidden_units = [5]
elif arch == 2:
	hidden_units = [5, 5]
elif arch == 3:
	hidden_units = [10]
elif arch == 4:
	hidden_units = [10, 10]
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# List for results
results = []
np.random.seed(seed)

# Run optimizations
for lam in lam_list:

	# Run experiment
	train_loss, val_loss, weights = run_iid_experiment(X_train, Y_train, X_val, Y_val, 
		nepoch, lr, lam, hidden_units, opt, penalty_groups, mbsize)
	
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

# Save results
with open('bike.out', 'wb') as f:
	pickle.dump(results, f)

for results_dict in results:
	print('show something interesting!')

# Plot validation loss curves
fig = plt.figure()
for i, result_dict in enumerate(results, 0):
	ax = fig.add_subplot(2, 4, i + 1)
	ax.plot(result_dict['val_loss'], color = 'orange')
	ax.set_title('lam = %f' % lam_list[i])
plt.show()

# Minimum validation loss
for result_dict in results:
	print(np.min(result_dict['val_loss']))

# Plot GC estimates
fig = plt.figure()
for i, result in enumerate(results, 0):
	ax = fig.add_subplot(len(lam_list), 1, i + 1)
	ax.imshow(result['GC_est'][np.newaxis], cmap = 'gray')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
plt.show()

fig = plt.figure()
for i, result in enumerate(results, 0):
	ax = fig.add_subplot(len(lam_list), 1, i + 1)
	ax.imshow(result['adj_GC_est'][np.newaxis], cmap = 'gray')
	ax.xaxis.set_visible(False)
	ax.yaxis.set_visible(False)
plt.show()

# Plot GC pathways
GC_pathways = np.zeros((len(results), len(input_groups)))
for i in range(len(results)):
	result_dict = results[i]
	GC_pathways[i, :] = result_dict['GC_est']
plt.semilogx(lam_list, GC_pathways)
plt.show()
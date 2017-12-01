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

# Parse command line arguments
mbsize = 32
lam_list = [1.0, 0.1, 0.01, 0.001, 0.0001]
opt = 'prox'
seed = 123
nepoch = 30000
arch = 1
lr = 0.005

# Prepare data
f1 = 'Data/Gene Expression/data.csv'
f2 = 'Data/Gene Expression/labels.csv'
df1 = pd.read_csv(f1, header = 0, sep = ',')
df2 = pd.read_csv(f2, header = 0, sep = ',')

data = df1.values[:, 1:].astype(float)
N, p_in = data.shape
label_names = ['PRAD', 'LUAD', 'BRCA', 'KIRC', 'COAD']
p_out = len(label_names)
labels = np.zeros((N, 1))
for i, label in enumerate(label_names):
	inds = df2['Class'].values == label
	labels[inds, :] = i

full_data = np.concatenate((data, labels), axis = 1)
np.random.seed(12345)
train, val, test = split_data(full_data, validation = 0.1, test = 0.1, shuffle = True)

Y_train = train[:, -1].astype(int)
Y_val = val[:, -1].astype(int)
X_train = train[:, :-1]
X_val = val[:, :-1]

penalty_groups = list(np.arange(0, p_in + 1))

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
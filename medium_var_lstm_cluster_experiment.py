import numpy as np
import pandas as pd
import argparse
import os
from itertools import product
import pickle
import shutil
import sys

from data_processing import *
from experiment import run_recurrent_experiment
from model_lstm import *

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
args = parser.parse_args()

nepoch = args.nepoch
lr = args.lr
opt_type = 'prox'
seed = args.seed
lam = args.lam

window_size = 100
stride_size = 10
truncation = 5

arch = 2

# Prepare filename
experiment_name = 'Experiments/LSTM/Medium VAR Encoding/expt'
experiment_name += '_lam=%e_nepoch=%d_lr=%e_seed-%d.out' % (lam, nepoch, lr, seed)

# Create directory, if necessary
if not os.path.exists('Experiments/LSTM/Medium VAR Encoding'):
     os.makedirs('Experiments/LSTM/Medium VAR Encoding')

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
filename = 'Data/medium_var.csv'
data = pd.read_csv(filename, dtype = float, header = 0, sep = ',')
data_values = data.values[:, 1:]

X_train, X_val = split_data(data_values, validation = 0.1)
Y_train = X_train[1:, :]
X_train = X_train[:-1, :]
Y_val = X_val[1:, :]
X_val = X_val[:-1, :]

p_in = X_val.shape[1]
p_out = Y_val.shape[1]

# Determine architecture
if arch == 1:
	hidden_size = 2
	hidden_layers = 1
elif arch == 2:
	hidden_size = 4
	hidden_layers = 1
elif arch == 3:
	hidden_size = 4
	hidden_layers = 2 
elif arch == 4:
	hidden_size = 6
	hidden_layers = 2
else:
	raise ValueError('arch must be in {1, 2, 3, 4}')

# Get model
torch.manual_seed(seed)
model = ParallelLSTMEncoding(p_in, p_out, hidden_size, hidden_layers, lr, opt_type, lam)

# Run experiment
train_loss, val_loss, weights_list, forecasts_train, forecasts_val = run_recurrent_experiment(model, X_train, Y_train, X_val, Y_val, 
	nepoch, window_size = window_size, stride_size = stride_size, truncation = truncation, predictions = True, loss_check = 10)

# Create GC estimate grid
GC_est = np.zeros((p_out, p_in))
for target in range(p_out):
	W = weights_list[target]
	GC_est[target, :] = np.linalg.norm(W, axis = 0, ord = 2)

# Save results
results_dict = {
	'lam': lam,
	'train_loss': train_loss,
	'val_loss': val_loss,
	'weights_list': weights_list,
	'GC_est': GC_est,
	'forecasts_train': forecasts_train,
	'forecasts_val': forecasts_val
}

# Save results
with open(experiment_name, 'wb') as f:
	pickle.dump(results_dict, f)

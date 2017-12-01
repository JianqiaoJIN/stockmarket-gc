import numpy as np
import argparse
import os
from itertools import product
import pickle
import shutil
import sys

from data_processing import *
from experiment import run_experiment
from model_iid import *

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', type = int, default = 1000, help = 'number of training epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'learning rate')
# parser.add_argument('--opt', type = str, default = 'momentum', help = 'optimization algorithm')
parser.add_argument('--lam', type = float, default = 0.1, help = 'lambda for weight decay')
parser.add_argument('--mbsize', type = int, default = None, help = 'minibatch size')
parser.add_argument('--seed', type = int, default = 12345, help = 'seed')
args = parser.parse_args()

nepoch = args.nepoch
lr = args.lr
opt = 'prox'
lam = args.lam
mbsize = args.mbsize
seed = args.seed

# nepoch = 5000
# lr = 0.001
# opt = 'prox'
# lam = 0.01
# mbsize = 32
# seed = 12345

# Prepare filename
if mbsize is None:
	batchsize = 0
else:
	batchsize = mbsize
experiment_name = 'Experiments/Genetics/Sparse/expt'
experiment_name += '_lam-%e_lr=%e_opt-%s_batch-%d_seed-%d_nepoch-%d.out' % (lam, lr, opt, batchsize, seed, nepoch)
print(experiment_name)

# Create directory, if necessary
if not os.path.exists('Experiments/Genetics/Sparse'):
     os.makedirs('Experiments/Genetics/Sparse')

# Verify that experiment doesn't exist
if os.path.isfile(experiment_name):
	print('Skipping experiment')
	sys.exit(0)

# Prepare data
train_filename = 'Data/Gene Expression/Experiment/train.csv'
val_filename = 'Data/Gene Expression/Experiment/validation.csv'

train = np.loadtxt(train_filename, delimiter = ',')
val = np.loadtxt(val_filename, delimiter = ',')

p_out = 5
p_in = train.shape[1] - 1
penalty_groups = list(np.arange(0, p_in + 1))

X_train = train[:, :-1] 
Y_train = train[:, -1]
X_val = val[:, :-1]
Y_val = val[:, -1]

# Determine architecture
hidden_units = [20]

# Prepare model
model = IIDEncoding(p_in, p_out, hidden_units, lr, opt, lam, penalty_groups, nonlinearity = 'sigmoid', task = 'classification')

# Run experiment
train_loss, val_loss, train_accuracy, val_accuracy, weights = run_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = mbsize, loss_check = 10)

# Save results
results = {
	'nepoch': nepoch,
	'lr': lr,
	'opt': opt,
	'lam': lam,
	'batchsize': batchsize,
	'seed': seed,
	'train_loss': train_loss,
	'train_accuracy': train_accuracy,
	'val_loss': val_loss,
	'val_accuracy': val_accuracy,
	'model': model
}

with open(experiment_name, 'wb') as f:
	pickle.dump(results, f)


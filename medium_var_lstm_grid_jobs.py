from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'medium_var_encoding_experiment_%s_%s' % (dstamp, tstamp)
jobfile = '%s.job' % jobname

lam_grid = [10.0, 7.5, 5.0, 2.5, 1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]
nepoch_grid = [2000]
seed_grid = [987]
lr_grid = [0.001]

BASECMD = 'python medium_var_lstm_cluster_experiment.py'

param_grid = product(lam_grid, nepoch_grid, seed_grid, lr_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, nepoch, seed, lr = param
		argstr = BASECMD
		argstr += ' --lam=%e' % lam
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --seed=%d' % seed
		argstr += ' --lr=%e' % lr

		f.write(argstr + '\n')

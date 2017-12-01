from itertools import product
import numpy as np
import time

dstamp = time.strftime('%Y%m%d')
tstamp = time.strftime('%H%M%S')

jobname = 'gene_experiment_%s_%s' % (dstamp, tstamp)
jobfile = '%s.job' % jobname

''' Jobs for ridge experiment '''

lam_grid = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
nepoch_grid = [5000]
seed_grid = [987, 876, 765, 654, 543]
lr_grid = [0.01, 0.005, 0.001, 0.0005, 0.0001]
opt_grid = ['momentum']
mbsize_grid = [None, 32]

BASECMD = 'python gene_ridge_experiment.py'

param_grid = product(lam_grid, nepoch_grid, seed_grid, lr_grid, opt_grid, mbsize_grid)

with open(jobfile, 'w') as f:
	for param in param_grid:
		lam, nepoch, seed, lr, opt, mbsize = param
		argstr = BASECMD
		argstr += ' --lam=%e' % lam
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --seed=%d' % seed
		argstr += ' --lr=%e' % lr
		argstr += ' --opt=%s' % opt
		if mbsize is not None:
			argstr += ' --mbsize=%d' % mbsize

		f.write(argstr + '\n')

''' Jobs for sparse experiment '''

lam_grid = [1.0, 0.5, 0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
nepoch_grid = [5000]
seed_grid = [987, 876, 765, 654, 543]
lr_grid = [0.01, 0.005, 0.001, 0.0005, 0.0001]
mbsize_grid = [None, 32]

BASECMD = 'python gene_sparse_experiment.py'

param_grid = product(lam_grid, nepoch_grid, seed_grid, lr_grid, opt_grid, mbsize_grid)

with open(jobfile, 'a') as f:
	for param in param_grid:
		lam, nepoch, seed, lr, opt, mbsize = param
		argstr = BASECMD
		argstr += ' --lam=%e' % lam
		argstr += ' --nepoch=%d' % nepoch
		argstr += ' --seed=%d' % seed
		argstr += ' --lr=%e' % lr
		if mbsize is not None:
			argstr += ' --mbsize=%d' % mbsize

		f.write(argstr + '\n')

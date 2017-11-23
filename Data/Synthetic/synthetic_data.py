'''
	The below code was adapted from a previous project in the Mode Lab
'''

from __future__ import division
import numpy as np
import pickle

def lorentz_96(forcing_constant, p, N, delta_t = 0.01, sd = 0.1, noise_add = 'global'):
	burnin = 4000
	N += burnin
	z = np.zeros((N, p))
	z[0, :] = np.random.normal(loc = 0, scale = 0.01, size = p)
	for t in range(1, N):
		for i in range(p):
			upone = (i + 1) % p
			grad = (z[t - 1, upone] - z[t - 1, i - 2]) * z[t - 1, i - 1] - z[t - 1, i] + forcing_constant
			z[t, i] = delta_t * grad + z[t - 1, i]
			if noise_add == 'step':
				z[t, i] += np.random.normal(loc = 0, scale = sd, size = 1)

	if noise_add == 'global':
		z += np.random.normal(loc = 0, scale = sd, size = (N, p))

	GC_on = np.zeros((p, p))
	for i in range(p):
		GC_on[i, i] = 1
		GC_on[i, (i - 1) % p] = 1
		GC_on[i, (i - 2) % p] = 1
		GC_on[i, (i + 1) % p] = 1

	return z[range(burnin, N), :], GC_on

def stationary_var(beta,p,lag,radius):
	bottom = np.hstack((np.eye(p * (lag-1)), np.zeros((p * (lag - 1), p))))  
	beta_tilde = np.vstack((beta,bottom))
	eig = np.linalg.eigvals(beta_tilde)
	maxeig = max(np.absolute(eig))
	not_stationary = maxeig >= radius
	return beta * 0.95, not_stationary

def var_model(sparsity, p, sd_beta, sd_e, N, lag):
	radius = 0.97
	min_effect = 1
	beta = np.random.normal(loc = 0, scale = sd_beta, size = (p, p * lag))
	beta[(beta < min_effect) & (beta > 0)] = min_effect
	beta[(beta > - min_effect) & (beta < 0)] = - min_effect

	GC_on = np.random.binomial(n = 1, p = sparsity, size = (p, p))
	for i in range(p):
		beta[i, i] = min_effect
		GC_on[i, i] = 1

	GC_lag = GC_on
	for i in range(lag - 1):
		GC_lag = np.hstack((GC_lag, GC_on))

	beta = np.multiply(GC_lag, beta)
	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))

	not_stationary = True
	while not_stationary:
		beta, not_stationary = stationary_var(beta, p, lag, radius)

	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta, GC_on

if __name__ == '__main__':

	X, GC = lorentz_96(5, 10, 1000)
	with open('lorentz_data.data', 'wb') as f:
		pickle.dump({'X': X, 'GC': GC}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	np.savetxt('../lorentz.csv', Y, delimiter = ',')

	X, beta, GC = var_model(0.2, 10, 1.0, 1.0, 1000, 1)
	with open('var_data.data', 'wb') as f:
		pickle.dump({'X': X, 'GC': GC, 'beta': beta}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	np.savetxt('../var.csv', Y, delimiter = ',')


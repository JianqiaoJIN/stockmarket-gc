'''
	The below code was adapted from a previous project in the Mode Lab
'''

from __future__ import division
import numpy as np
import pickle
from scipy.special import logsumexp

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

def long_lag_var_model(sparsity, p, sd_beta, sd_e, N, lag = 20):
	radius = 0.97
	min_effect = 1
	GC_on = np.random.binomial(n = 1, p = sparsity, size = (p, p))
	np.fill_diagonal(GC_on, 0.0)
	GC_lag = np.zeros((p, p * lag))
	GC_lag[:, range(0, p)] = np.eye(p)
	GC_lag[:, range(p * (lag - 1), p * lag)] = GC_on

	beta = np.random.normal(loc = 0, scale = sd_beta, size = (p, p * lag))
	beta[(beta < min_effect) & (beta > 0)] = min_effect
	beta[(beta > min_effect) & (beta < 0)] = - min_effect
	beta = np.multiply(beta, GC_lag)

	not_stationary = True
	while not_stationary:
		beta, not_stationary = stationary_var(beta, p, lag, radius)

	errors = np.random.normal(loc = 0, scale = sd_e, size = (p, N))

	X = np.zeros((p, N))
	X[:, range(lag)] = errors[:, range(lag)]
	for i in range(lag, N):
		X[:, i] = np.dot(beta, X[:, range(i - lag, i)].flatten(order = 'F')) + errors[:, i]

	return X.T, beta, np.maximum(GC_on, np.eye(p))

"""function generates multivariate time series data from a 
sparse HMM model where there is sparsity in the HMM transition function
inputs:
p - dimensional of the time series
N - length of time series
num_states - #of states of the HMM
sd_e - standard deviation of the error distribution for HMM
sparsity - sparsity proportion
tau - temperature parameter
you might need to mess with the sd_e parameter, as a bigger value will 
make dependence on the past more important for inference of the future.
"""
def generate_data_hmm(p, N, num_states = 3, sd_e = 0.1, sparsity = 0.2, tau = 2):
	Z_sig = 0.3
	Z = np.zeros((p, p, num_states, num_states))
	GC_on = np.random.binomial(1, sparsity, p * p).reshape(p,p)
	for i in range(p):
		GC_on[i,i] = 1
	mu = np.random.uniform(low = -5.0, high = 5.0, size = (p, num_states))
	for i in range(p):
		for j in range(p):
			if GC_on[i,j]:
				Z[i,j,:,:] = Z_sig * np.random.randn(num_states,num_states)

    # generate state sequence
	L = np.zeros((N,p)).astype(int)
	for t in range(1,N):
		for i in range(p):
			switch_prob = np.zeros(num_states)
			for j in range(p):
				switch_prob += Z[i,j,L[t-1,j],:]
			switch_prob = switch_prob * tau
			switch_prob = np.exp(switch_prob - logsumexp(switch_prob))
			L[t,i] = np.nonzero(np.random.multinomial(1, switch_prob))[0][0]

    # generate outputs from state sequence
	X = np.zeros((N,p))
	for i in range(N):
		for j in range(p):
			X[i,j] = sd_e * np.random.randn(1) + mu[j,L[i,j]]

	return X, L, GC_on

if __name__ == '__main__':

	X, GC = lorentz_96(5, 10, 1000)
	# with open('lorentz_data.data', 'wb') as f:
	# 	pickle.dump({'X': X, 'GC': GC}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	# np.savetxt('../lorentz.csv', Y, delimiter = ',')

	X, beta, GC = var_model(0.2, 10, 1.0, 1.0, 1000, 1)
	# with open('var_data.data', 'wb') as f:
	# 	pickle.dump({'X': X, 'GC': GC, 'beta': beta}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	# np.savetxt('../var.csv', Y, delimiter = ',')

	X, beta, GC = long_lag_var_model(0.2, 10, 1.0, 1.0, 1000, lag = 20)
	with open('long_var_data.data', 'wb') as f:
		pickle.dump({'X': X, 'GC': GC, 'beta': beta}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	np.savetxt('../long_var.csv', Y, delimiter = ',')

	X, beta, GC = long_lag_var_model(0.2, 10, 1.0, 1.0, 1000, lag = 3)
	with open('medium_var_data.data', 'wb') as f:
		pickle.dump({'X': X, 'GC': GC, 'beta': beta}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	np.savetxt('../medium_var.csv', Y, delimiter = ',')	

	X, L, GC = generate_data_hmm(10, 1000, num_states = 3, sd_e = 0.1, sparsity = 0.2, tau = 2)
	with open('hmm_data.data', 'wb') as f:
		pickle.dump({'X': X, 'GC': GC, 'L': L}, f, pickle.HIGHEST_PROTOCOL)

	Y = np.zeros((1001, 11))
	Y[1:, 1:] = X
	Y[1:, 0] = np.arange(1, 1001)
	Y[0, :] = np.arange(1, 12)
	np.savetxt('../hmm.csv', Y, delimiter = ',')

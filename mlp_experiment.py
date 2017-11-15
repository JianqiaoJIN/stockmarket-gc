from math import ceil
import numpy as np

from model import ParallelMLP

def run_experiment(X_train, Y_train, X_val, Y_val, lag, nepoch, lr, lam, penalty_type, hidden_units, opt, mbsize = None, verbose = True, loss_check = 100):
	# Get data size
	T_train, p_in = X_train.shape
	p_out = Y_train.shape[1]

	# Batch parameters
	minibatches = not mbsize is None
	if minibatches:
		n_batches = ceil(T_train / mbsize)

	# Create MLP model
	model = ParallelMLP(p_in, int(p_in / lag), lag, p_out, hidden_units, lr, opt, lam, penalty_type)

	# Prepare for training
	nchecks = int(nepoch / loss_check)
	train_loss = np.zeros((nchecks, p_out))
	val_loss = np.zeros((nchecks, p_out))
	counter = 0

	# Begin training
	for epoch in range(nepoch):

		# Run training step
		if minibatches:
			for batch in range(n_batches - 1):
				x_batch = X_train[range(i * mbsize, (i + 1) * mbsize), :]
				y_batch = Y_train[range(i * mbsize, (i + 1) * mbsize), :]
				model.train(x_batch, y_batch)

			x_batch = X_train[range(i * mbsize, T_train), :]
			y_batch = Y_train[range(i * mbsize, N_train), :]
			model.train(x_batch, y_batch)

		else:
			model.train(X_train, Y_train)

		# Check progress
		if epoch % loss_check == 0:
			# Save results
			train_loss[counter, :] = model.calculate_mse(X_train, Y_train)
			val_loss[counter, :] = model.calculate_mse(X_val, Y_val)

			# Print results
			if verbose:
				print('----------')
				print('epoch %d' % epoch)
				print('train loss = %e' % train_loss[counter, 0])
				print('val loss = %e' % val_loss[counter, 0])
				print('----------')

			counter += 1

	weights = model.get_weights()

	if verbose:
		print('Done training')

	return train_loss, val_loss, weights


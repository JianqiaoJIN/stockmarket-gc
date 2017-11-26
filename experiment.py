import numpy as np
from math import ceil

def run_experiment(model, X_train, Y_train, X_val, Y_val, nepoch, mbsize = None, verbose = True, loss_check = 100, predictions = False):
	# Batch parameters
	minibatches = not mbsize is None
	if minibatches:
		n_batches = ceil(N_train / mbsize)

	# Prepare for training
	nchecks = int(nepoch / loss_check)
	if len(Y_train.shape) == 1:
		d_out = 1
	else:
		d_out = Y_train.shape[1]
	train_loss = np.zeros((nchecks, d_out))
	val_loss = np.zeros((nchecks, d_out))
	counter = 0

	# Begin training
	for epoch in range(nepoch):

		# Run training step
		if minibatches:
			for i in range(n_batches - 1):
				x_batch = X_train[range(i * mbsize, (i + 1) * mbsize), :]
				y_batch = Y_train[range(i * mbsize, (i + 1) * mbsize), :]
				model.train(x_batch, y_batch)

			x_batch = X_train[range((n_batches - 1) * mbsize, N_train), :]
			y_batch = Y_train[range((n_batches - 1) * mbsize, N_train), :]
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

	if predictions:
		return train_loss, val_loss, weights, model.predict(X_train), model.predict(X_val)
	else:
		return train_loss, val_loss, weights

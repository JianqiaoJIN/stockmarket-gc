import pandas as pd
import numpy as np
import os

data_dir = '2002-2006'

sectors = ['tech', 'finance', 'health', 'industrials', 'consumer']
tickers_list = [
	['AAPL', 'ORCL', 'IBM', 'AMZN', 'MSFT'],
	['GS', 'MS', 'JPM', 'AXP', 'TRV'],
	['MCK', 'A', 'PFE', 'AMGN', 'CVS'],
	['GE', 'DE', 'GD', 'CAT', 'UTX'],
	['EL', 'UL', 'NKE', 'HAS', 'SNE']
]

for sector, tickers in zip(sectors, tickers_list):

	tickers_vol = [tick + ' Volume' for tick in tickers]
	cols = [p for v in zip(tickers, tickers_vol) for p in v]

	data = pd.DataFrame(columns = cols)

	for ticker, v_label in zip(tickers, tickers_vol):
		filename = os.path.join(data_dir, ticker + '.csv')
		df = pd.read_csv(filename)
		data[ticker] = df['Close']
		data[v_label] = df['Volume']

	# Whiten volume data
	for v_label in tickers_vol:
		v = data[v_label].values
		v_tilde = v - np.mean(v)
		data[v_label] = v_tilde / np.max(np.abs(v_tilde))

	# Closing prices
	outname = 'volume_price_' + sector + '.csv'
	data.to_csv(outname)

	# Returns
	returns = pd.DataFrame(columns = cols)
	for ticker, v_label in zip(tickers, tickers_vol):
		returns[ticker] = np.diff(data[ticker].values, axis = 0) / data[ticker].values[:-1]
		returns[v_label] = data[v_label].values[1:]
	outname = 'volume_price_returns_' + sector + '.csv'
	returns.to_csv(outname)

	# Log returns
	for ticker in tickers:
		returns[ticker] = np.log(returns[ticker] + 1)
	outname = 'volume_price_log_returns_' + sector + '.csv'
	returns.to_csv(outname)
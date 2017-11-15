import pandas as pd
import numpy as np
import argparse
import os

# Parse location of ticker symbols
parser = argparse.ArgumentParser()
parser.add_argument('--tickers', type = str, default = None, help = 'file containing list of tickers')
parser.add_argument('--dir', type = str, default = None, help = 'directory containing raw data')
args = parser.parse_args()

tickers_name = args.tickers
data_dir = args.dir

# Read tickers
with open(tickers_name, 'r') as f:
	tickers = [ticker.strip() for ticker in f.readlines()]

# Create empty DataFrame
closing = pd.DataFrame(columns = tickers)

# Populate columns of DataFrame
for ticker in tickers:
	filename = os.path.join(data_dir, ticker + '.csv')
	df = pd.read_csv(filename)
	closing[ticker] = df['Close']

outname = 'closing_' + data_dir + '.csv'
closing.to_csv(outname)

# Take diff
returns = pd.DataFrame(columns = tickers)
for ticker in tickers:
	returns[ticker] = np.diff(closing[ticker].values, axis = 0) / closing[ticker].values[:-1]
outname = 'returns_' + data_dir + '.csv'
returns.to_csv(outname)

# Take log
for ticker in tickers:
	returns[ticker] = np.log(returns[ticker] + 1)
outname = 'log_returns_' + data_dir + '.csv'
returns.to_csv(outname)
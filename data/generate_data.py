import numpy as np
import pandas as pd
import os

#  CONFIG 
np.random.seed(42)
n_stocks = 30
n_days = 365

# randomize starting prices between 20 and 300
base_prices = np.random.uniform(20, 300, size=n_stocks)

# group structure: 3 groups of 10 (within-group correlated, across-group less correlated)
group_corr = 0.6    # within group correlation
cross_corr = 0.1    # between groups
daily_vol = 0.01    # ~1% daily vol → ~16% annualized
daily_mu = 0.0004   # ~10% annual drift

#  BUILD CORRELATION MATRIX 
corr = np.full((n_stocks, n_stocks), cross_corr)
for g in range(3):
    i0, i1 = g * 10, (g + 1) * 10
    corr[i0:i1, i0:i1] = group_corr
np.fill_diagonal(corr, 1.0)

# convert to covariance
cov = corr * (daily_vol ** 2)

#  SAMPLE MULTIVARIATE LOG RETURNS 
L = np.linalg.cholesky(cov)
z = np.random.randn(n_days, n_stocks)
log_rets = z @ L.T + daily_mu  # correlated log returns

#  CONVERT TO PRICE SERIES 
# geometric Brownian motion with different base prices
prices = np.exp(np.cumsum(log_rets, axis=0)) * base_prices
dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)
tickers = [f"STOCK{i+1:02d}" for i in range(n_stocks)]
prices_df = pd.DataFrame(prices, index=dates, columns=tickers).round(2)

#  SAVE TO CSV 
csv_path = os.path.join(os.getcwd(), "fake_correlated_stocks.csv")
prices_df.to_csv(csv_path, index_label="Date")

csv_path

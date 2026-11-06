import pandas as pd
import numpy as np

def calculate_metrics(dataset):

  dataset["Return"] = dataset["Close"].pct_change()
  dataset["SM5"] = dataset["Return"].rolling(window = 5).mean()
  dataset["SM20"] = dataset["Return"].rolling(window = 20).mean()
  dataset["Volatility_10d"] = dataset["Return"].rolling(window = 20).std()
  dataset["Volume / Vol20"] = dataset["Volume"] / dataset["Volume"].rolling(window = 20).mean()
  dataset = dataset.drop(columns = "Volume")
  conditions = [dataset["Return"].shift(-1) > 0, dataset["Return"].shift(-1) < 0, pd.isna(dataset["Return"].shift(-1))
  ]
  choices = [1, 0, np.nan]
  dataset["Target"] = np.select(condlist = conditions, choicelist = choices)
  dataset = dataset.drop(columns = "Close")
  dataset["Momentum 12-1"] = dataset["Return"].rolling(window = 138).mean()
  dataset = dataset.dropna()
  dataset["Target"] = dataset["Target"].astype(int)
  

  return dataset

def calculate_rsi(prices, period=14):
  delta = prices.diff()
  gain = delta.where(delta > 0, 0.0)
  loss = -delta.where(delta < 0, 0.0)
  avg_gain = gain.rolling(window=period, min_periods=period).mean()
  avg_loss = loss.rolling(window=period, min_periods=period).mean()
  rs = avg_gain / avg_loss
  rsi =  100 - (100 / (1 + rs))
  return rsi


def calculate_bollinger_pct_b(prices, period=20, num_std=2):
  middle_band = prices.rolling(window=period).mean()
  std = prices.rolling(window=period).std()
  upper_band = middle_band + (num_std * std)
  lower_band = middle_band - (num_std * std)
  pct_b = (prices - lower_band) / (upper_band - lower_band)
  return pct_b

def all_metrics_together(dataset, prices):

  dataset = calculate_metrics(dataset)
  rsi = calculate_rsi(prices, period=14)
  pct_b = calculate_bollinger_pct_b(prices, period=20, num_std=2)
  dataset["RSI"] = rsi
  dataset["PCTB"] = pct_b
  dataset = dataset.reset_index(drop= True)
  dataset = dataset.dropna()
  return dataset
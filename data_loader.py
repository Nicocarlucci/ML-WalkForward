import yfinance as yf
import pandas as pd


def download_data(stock, start_date, end_date):
    prices = yf.download(stock, start = start_date, end = end_date, interval = "1d")

    if isinstance(prices.columns, pd.MultiIndex):
        # Estraiamo direttamente Close e Volume appiattendo il MultiIndex se presente
        close_series = prices[('Close', stock)] if ('Close', stock) in prices.columns else prices['Close'].iloc[:, 0]
        volume_series = prices[('Volume', stock)] if ('Volume', stock) in prices.columns else prices['Volume'].iloc[:, 0]
        
        prices = pd.DataFrame({
            "Close": close_series,
            "Volume": volume_series
        })
    else:
        
        prices = prices[["Close", "Volume"]].copy()

    dataset = prices[["Close", "Volume"]].copy()
    prices = dataset["Close"]

    return dataset, prices

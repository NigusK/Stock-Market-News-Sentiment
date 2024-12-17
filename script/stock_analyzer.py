import yfinance as yf
import talib as ta
import pandas as pd
import numpy as np
# import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt import risk_models
# from pypfopt import expected_returns

class StockAnalyzer:
    # def __init__(self, ticker, start_date, end_date):
    #     self.ticker = ticker
    #     self.start_date = start_date
    #     self.end_date = end_date
    
    def retrieve_stock_data(self,ticker, start_date, end_date):
        return yf.download(ticker, start=start_date, end=end_date)
    
    
    def get_day_range(self, df, ticker):
        # Split the date column into date_only and time_only
        df['date_only'] = df['date'].apply(lambda x: x.split()[0])
        df['time_only'] = df['date'].apply(lambda x: x.split()[1].split('+')[0].split('-')[0])

        # Parse 'date_only' to datetime format
        df['date_only'] = pd.to_datetime(df['date_only'])

        # Use pd.to_datetime with a specified format for time_only
        df['time_only'] = pd.to_datetime(df['time_only'], format='%H:%M:%S').dt.time

        # Filter headlines for the given ticker
        filtered_headline = df[df['stock'] == ticker]
        
        # Find min and max dates
        max_date = filtered_headline['date_only'].max()
        min_date = filtered_headline['date_only'].min()

        print("Max date:", max_date)
        print("Min date:", min_date)

        return min_date,max_date
    
    def filter_stock_data(self,data):
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        data['Date'] = pd.to_datetime(data['Date'])
        
        filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]
        return filtered_data
    
    def calculate_moving_average(self, data, window_size):
        return ta.SMA(data, timeperiod=window_size)

    def calculate_technical_indicators(self, df,ticker):
        closing=df['Close'][ticker]
        # Calculate various technical indicators
        df.loc[:, 'SMA'] = ta.SMA(closing, timeperiod=20)
        df.loc[:,'RSI'] = ta.RSI(closing, timeperiod=14)
        df.loc[:,'EMA'] = ta.EMA(closing, timeperiod=20)
        macd, macd_signal, _ = ta.MACD(closing)
        df.loc[:,'MACD'] = macd
        df.loc[:,'MACD_Signal'] = macd_signal
        return df
   
    def plot_technical_indicators(self,data):
        fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        fig.suptitle('Stock Analysis with Technical Indicators', fontsize=16)

        # Plot stock price and SMA
        axs[0].plot(data.index, data['Close'], label='Close', color='blue')
        axs[0].plot(data.index, data['SMA'], label='SMA', color='orange')
        axs[0].set_title('Stock Price and SMA')
        axs[0].legend(loc='upper left')

        # Plot RSI
        axs[1].plot(data.index, data['RSI'], label='RSI', color='green')
        axs[1].axhline(70, color='red', linestyle='--', label='Overbought')
        axs[1].axhline(30, color='blue', linestyle='--', label='Oversold')
        axs[1].set_title('Relative Strength Index (RSI)')
        axs[1].legend(loc='upper left')

        # Plot EMA
        axs[2].plot(data.index, data['Close'], label='Close', color='blue')
        axs[2].plot(data.index, data['EMA'], label='EMA', color='red')
        axs[2].set_title('Stock Price and EMA')
        axs[2].legend(loc='upper left')

        # Plot MACD
        axs[3].plot(data.index, data['MACD'], label='MACD', color='purple')
        axs[3].plot(data.index, data['MACD_Signal'], label='MACD Signal', color='orange')
        axs[3].set_title('Moving Average Convergence Divergence (MACD)')
        axs[3].legend(loc='upper left')

        # Formatting
        for ax in axs:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            ax.grid(True)

        plt.xticks(rotation=90)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    # def calculate_portfolio_weights(self, tickers, start_date, end_date):
    #     data = yf.download(tickers, start=start_date, end=end_date)['Close']
    #     mu = expected_returns.mean_historical_return(data)
    #     cov = risk_models.sample_cov(data)
    #     ef = EfficientFrontier(mu, cov)
    #     weights = ef.max_sharpe()
    #     weights = dict(zip(tickers, weights.values()))
    #     return weights

    # def calculate_portfolio_performance(self, tickers, start_date, end_date):
    #     data = yf.download(tickers, start=start_date, end=end_date)['Close']
    #     mu = expected_returns.mean_historical_return(data)
    #     cov = risk_models.sample_cov(data)
    #     ef = EfficientFrontier(mu, cov)
    #     weights = ef.max_sharpe()
    #     portfolio_return, portfolio_volatility, sharpe_ratio = ef.portfolio_performance()
    #     return portfolio_return, portfolio_volatility, sharpe_ratio



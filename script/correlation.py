import pandas as pd
import numpy as np
# import plotly.express as px
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

class CorrelationAnalyzer:
    def correlation(self,sentiment,hist_data,ticker):
        hist_data = hist_data.drop([0, 1]).reset_index(drop=True)
        hist_data.rename(columns={'Price': 'Date'}, inplace=True)
        # Convert 'Date' to datetime
        hist_data['Date'] = pd.to_datetime(hist_data['Date'], errors='coerce')
        #Extract company's sentiment data
        sentiment_data=sentiment[sentiment['stock']==ticker]
        # Group by 'date_only' and calculate the mean of 'compound'
        result_df = sentiment_data.groupby('date_only')['compound'].mean().reset_index()

        # Rename columns
        result_df.columns = ['Date', 'Sentiment']
        result_df['Date'] = pd.to_datetime(result_df['Date'], errors='coerce')
        
        
        #combining result_df with stock data which have matching dates
        aggregate_data = pd.merge(hist_data, result_df, left_on='Date', right_on='Date', how='inner')
        # Convert 'Close' to numeric
        aggregate_data['Close'] = pd.to_numeric(aggregate_data['Close'], errors='coerce')

        # Drop rows with NaN in 'Close' after conversion
        aggregate_data.dropna(subset=['Close'], inplace=True)
        
        #calculate daily returns
        aggregate_data['Daily Returns'] = aggregate_data['Close'].pct_change()
        
        # Calculate the Pearson correlation coefficient
        cleaned_df = aggregate_data.dropna(subset=['Sentiment', 'Daily Returns'])
        pearson_correlation, _ = pearsonr(cleaned_df['Sentiment'], cleaned_df['Daily Returns'])

        print("Pearson correlation coefficient:", pearson_correlation)
        
        return aggregate_data
        
    def sentiment_vs_daily_return_plot(self,df):
        # Ensure the 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'])
        # Plot Sentiment and Daily Returns
        fig, ax1 = plt.subplots(figsize=(14, 7))

        # Plot Sentiment on the first y-axis
        ax1.plot(df['Date'], df['Sentiment'], color='blue', label='Sentiment')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Sentiment', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Create a second y-axis for Daily Returns
        ax2 = ax1.twinx()
        ax2.plot(df['Date'], df['Daily Returns'], color='red', label='Daily Returns')
        ax2.set_ylabel('Daily Returns', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Set the title of the plot
        plt.title('Sentiment vs. Daily Returns Over Time')

        # Improve the layout
        fig.tight_layout()

        # Show the plot
        plt.show()
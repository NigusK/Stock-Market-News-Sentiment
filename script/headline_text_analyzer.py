import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

class HeadlineAnalyzer:
    def time_conversion(self,df):
        df['date_only'] = df['date'].apply(lambda x: x.split()[0])
        df['time_only'] = df['date'].apply(lambda x: x.split()[1].split('+')[0].split('-')[0])
        
        df['date_only']=pd.to_datetime(df['date_only'])
        df['time_only']=pd.to_datetime(df['time_only']).dt.time
        
        return df
    def headline_length_plot(self,df):
        df['headline_length'] = df['headline'].str.len()
        
        # Plotting the distribution of headline lengths
        plt.figure(figsize=(10, 6))
        plt.hist(df['headline_length'], bins=20, color='skyblue')
        plt.title('Distribution of Headline Lengths')
        plt.xlabel('Headline Length')
        plt.ylabel('Frequency')
        plt.show()
    def publisher_count(self,df):
        publisher_counts = df['publisher'].value_counts()
        print(publisher_counts)
        
        # Plotting the most active publishers
        plt.figure(figsize=(10, 6))
        publisher_counts.head(10).plot(kind='bar', color='coral')
        plt.title('Top 10 Most Active Publishers')
        plt.xlabel('Publisher')
        plt.ylabel('Number of Articles')
        plt.show()
    
    def publication_freq_plot(self,df):
        # Extract day of the week, hour, and month
        df['day_of_week'] = df['date_only'].dt.day_name()
        df['hour'] = df['time_only'].apply(lambda x: x.hour if pd.notnull(x) else None)
        df['month'] = df['date_only'].dt.month_name()

        # Count articles by day of the week
        day_of_week_counts = df['day_of_week'].value_counts()

        # Count articles by hour
        hour_counts = df['hour'].value_counts().sort_index()

        # Count articles by month
        month_counts = df['month'].value_counts().reindex([
            'January', 'February', 'March', 'April', 'May', 'June', 
            'July', 'August', 'September', 'October', 'November', 'December'
        ], fill_value=0)

        # Plotting publication frequency
        plt.figure(figsize=(16, 15))

        # Plot day of the week
        plt.subplot(3, 1, 1)
        day_of_week_counts.plot(kind='bar', color='lightgreen')
        plt.title('Articles Published by Day of the Week')
        plt.xlabel('Day of the Week')
        plt.ylabel('Number of Articles')

        # Plot hour of the day
        plt.subplot(3, 1, 2)
        hour_counts.plot(kind='bar', color='skyblue')
        plt.title('Articles Published by Hour of the Day')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')

        # Plot month of the year
        plt.subplot(3, 1, 3)
        month_counts.plot(kind='bar', color='salmon')
        plt.title('Articles Published by Month')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()
    
    def sentiment_analysis(self,data):
        sentiment_data=data.copy()
        sia = SentimentIntensityAnalyzer()
        # Calculate the sentiment of the headlines
        data.loc[:,'compound']=data['headline'].apply(lambda x: sia.polarity_scores(text=x)['compound'] )
        
        def categorize_sentiment(score):
            if score >= 0.6:
                return 'Very Positive'
            elif score >= 0.2:
                return 'Positive'
            elif score > -0.2:
                return 'Neutral'
            elif score > -0.6:
                return 'Negative'
            else:
                return 'Very Negative'

        # Apply categorization
        data.loc[:,'sentiment'] = data['compound'].apply(categorize_sentiment)
        
        # Count of each sentiment
        sentiment_counts = data['sentiment'].value_counts()
        print(sentiment_counts)
        
        # Plotting sentiment distribution
        plt.figure(figsize=(8, 6))
        sentiment_counts.plot(kind='bar', color=['lightgreen', 'lightcoral', 'lightgrey'])
        plt.title('Sentiment Distribution of Headlines')
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Headlines')
        plt.show()
        
    def time_series_analysis(self,df):
        # Count headlines per day
        daily_headlines = df.groupby('date_only').size()

        # Create a DataFrame for time series analysis
        time_series_data = pd.DataFrame({'Date': daily_headlines.index, 'Headlines_Count': daily_headlines.values})
        time_series_data.set_index('Date', inplace=True)
        
        decomposition = seasonal_decompose(time_series_data['Headlines_Count'], model='additive', period=150)

        # Plotting the components
        plt.figure(figsize=(14, 20))

        plt.subplot(411)
        plt.plot(time_series_data['Headlines_Count'], label='Original', color='blue')
        plt.legend(loc='upper left')

        plt.subplot(412)
        plt.plot(decomposition.trend, label='Trend', color='orange')
        plt.legend(loc='upper left')

        plt.subplot(413)
        plt.plot(decomposition.seasonal, label='Seasonal', color='green')
        plt.legend(loc='upper left')

        plt.subplot(414)
        plt.plot(decomposition.resid, label='Residual/Irregular', color='red')
        plt.legend(loc='upper left')

        plt.tight_layout()
        plt.show()
        
    def hourly_distribution(self,df):
        df['hour'] = df['time_only'].apply(lambda x: x.hour if pd.notnull(x) else None)

        # Count the number of articles published in each hour
        hourly_distribution = df['hour'].value_counts().sort_index()

        # Plot the hourly distribution of article publications
        plt.figure(figsize=(10, 6))
        hourly_distribution.plot(kind='bar', color='skyblue')
        plt.title('Hourly Distribution of Article Publications')
        plt.xlabel('Hour of the Day')
        plt.ylabel('Number of Articles')
        plt.show()

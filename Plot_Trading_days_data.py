import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np

def get_trading_days_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('Date', inplace=True)
    return df

trading_days_file_name = 'XX_daily_data.csv'
trading_days_data = get_trading_days_data(trading_days_file_name)
print(trading_days_data.head())

dates = trading_days_data.index.tolist()
datetime_objs = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in dates]

# Plot data
plt.figure(figsize=(12, 6))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Stemmed_Positive_Sentiment']), np.ones(ma_window)/ma_window, mode='valid')
moving_average_returns = np.convolve(list(trading_days_data['Stemmed_Positive_Sentiment']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Stemmed_Positive_Sentiment'], label='Positive Sentiment', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Positive Sentiment Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit (US Dollars)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.savefig('Plots/Positive_Sentiment_Over_Time.png', bbox_inches='tight')
plt.show()

# Plot data
plt.figure(figsize=(12, 6))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Stemmed_Negative_Sentiment']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Stemmed_Negative_Sentiment'], label='Negative Sentiment', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Negative Sentiment Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit (US Dollars)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.savefig('Plots/Negative_Sentiment_Over_Time.png', bbox_inches='tight')
plt.show()

# Plot data
plt.figure(figsize=(12, 6))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Media_Volume']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Media_Volume'], label='Media Volume', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Media Volume Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit (US Dollars)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.savefig('Plots/Media_Volume_Over_Time.png', bbox_inches='tight')
plt.show()
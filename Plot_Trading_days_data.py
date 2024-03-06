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
plt.plot(datetime_objs, trading_days_data['Stemmed_Positive_Sentiment'], label='Stemmed Positive Sentiment', color='#2980b9', linewidth=1)
plt.plot(datetime_objs, trading_days_data['Stemmed_Negative_Sentiment'], label='Stemmed Negative Sentiment', color='#e74c3c', linewidth=1)
#plt.plot(datetime_objs, trading_days_data['Returns'], label='Returns', color='#27ae60', linewidth=1)
#plt.plot(datetime_objs, trading_days_data['Media_Volume'], label='Media Volume', color='#27ae60', linewidth=1)
plt.title('Trading Profits Using VAR with Sentiment Vs Buy-and-Hold Strategy', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit (US Dollars)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.savefig('Plots/Trading_Strategy_Profits.png', bbox_inches='tight')
plt.show()
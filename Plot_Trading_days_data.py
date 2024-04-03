import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from collections import defaultdict
import statistics
import pandas as pd
import numpy as np
import pickle

# Class containing info about each article
class Article:
    def __init__(self, date, body, stemmed_text_body, source, headline, pos_sentiment, neg_sentiment, 
                 stemmed_text_pos_sentiment, stemmed_text_neg_sentiment):
        self.date = date
        self.body = body
        self.stemmed_text_body = stemmed_text_body
        self.source = source
        self.headline = headline
        self.pos_sentiment = pos_sentiment
        self.neg_sentiment = neg_sentiment
        self.stemmed_text_pos_sentiment = stemmed_text_pos_sentiment
        self.stemmed_text_neg_sentiment = stemmed_text_neg_sentiment

def get_trading_days_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('Date', inplace=True)
    return df

def convert_to_zscore(returns):
    z_score_returns = []
    
     # Calculate the mean & standard deviations
    mean = statistics.mean(returns)
    std_dev = statistics.stdev(returns)
    
    # Convert sentiments to Z-scores
    for daily_return in returns:
        z_score_returns.append((daily_return - mean) / std_dev)
        
    return z_score_returns

def get_media_vol(articles):
    daily_media_volume = {}
    for article in articles:
        if article.date in daily_media_volume:
            daily_media_volume[article.date] = daily_media_volume[article.date]+1
        else: daily_media_volume[article.date] = 1
    return daily_media_volume

trading_days_file_name = 'Aggregated_Time_Series.csv'
trading_days_data = get_trading_days_data(trading_days_file_name)
print(trading_days_data.head())

dates = trading_days_data.index.tolist()
datetime_objs = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in dates]

# Plot data
plt.figure(figsize=(16, 4))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Positive_Sentiment']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Positive_Sentiment'], label='Positive Sentiment', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Positive Sentiment Over Time', fontsize=18, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 13})
plt.xlabel('Date', fontsize=15, fontname='Times New Roman')
plt.ylabel('Standard Deviations', fontsize=15, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
#plt.savefig('Plots/Positive_Sentiment_Over_Time.png', bbox_inches='tight')
#plt.show()

# Plot data
plt.figure(figsize=(16, 4))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Negative_Sentiment']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Negative_Sentiment'], label='Negative Sentiment', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Negative Sentiment Over Time', fontsize=18, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 13})
plt.xlabel('Date', fontsize=15, fontname='Times New Roman')
plt.ylabel('Standard Deviations', fontsize=15, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('Plots/Negative_Sentiment_Over_Time.png', bbox_inches='tight')
#plt.show()

# Plot data
plt.figure(figsize=(16, 4))
# Calculate 60-period moving average
ma_window = 60
moving_average_sentiment = np.convolve(list(trading_days_data['Media_Volume']), np.ones(ma_window)/ma_window, mode='valid')
plt.plot(datetime_objs, trading_days_data['Media_Volume'], label='Article Count', color='#2980b9', linewidth=1)
plt.plot(datetime_objs[ma_window//2:-ma_window//2], moving_average_sentiment[1:], label='60-day Moving Average', color='#e74c3c', linewidth=1)
plt.title('Daily Article Count Over Time', fontsize=18, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.ylim(0, 60)
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 13})
plt.xlabel('Date', fontsize=15, fontname='Times New Roman')
plt.ylabel('Article Count (Articles)', fontsize=15, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.savefig('Plots/Media_Volume_Over_Time.png', bbox_inches='tight')
#plt.show()
plt.close()


######################################
# Get article count breakdown
articles_backup_path = 'Article_Data/Articles_backup.pkl'
with open(articles_backup_path, 'rb') as file:
    articles = pickle.load(file)
    print(f"Loaded {len(articles)} articles from backup file.")
        
media_vol = get_media_vol(articles)
yearly_counts = defaultdict(int)

# Aggregate the counts for each year
for date in media_vol:
    year = date.year
    yearly_counts[year] += media_vol[date]

# Print the total counts for each year
for year in range(2003, 2023 + 1):
    print(f'{yearly_counts[year]} & ', end='', flush=True)
    
# Create a bar chart
years = list(yearly_counts.keys())
total_counts = list(yearly_counts.values())
plt.figure(figsize=(15, 4.5))
plt.bar(years, total_counts, width=.7, color='#2980b9')
plt.xlabel('Year', fontsize=17, fontname='Times New Roman')
plt.ylabel('Yearly Article Count', fontsize=17, fontname='Times New Roman')
plt.xticks(range(min(years), max(years)+1, 2))
plt.title('Article Count Yearly Breakdown', fontsize=18, fontfamily='serif')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=12)

######################################
# Get monthly article count breakdown
monthly_counts = defaultdict(int)

# Aggregate the counts for each month
for date in media_vol:
    month = date.month
    year = date.year
    monthly_counts[(month,month)] += media_vol[date]

# Print the total counts for each month
for year in range(2003, 2023 + 1):
    for month in range(1, 12+1):
        if month == 1:
            print(f'\\\\ \n\\textbf{{{year}}} & ', end='', flush=True)
        else: print(' & ', end='', flush=True)
        print(f"{monthly_counts[(month,month)]}", end='', flush=True)
        if month == 12:
            print(f" & {yearly_counts[year]}", end='', flush=True)
    
print("SUM", sum(list(monthly_counts.values())))
######################################
    

plt.figure(figsize=(13, 5))
plt.plot(datetime_objs, list(trading_days_data['Returns']), label='RYAAY Returns', color='#2980b9', linewidth=1)
plt.title('RYAAY Returns Over Time', fontsize=18, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 13})
plt.xlabel('Date', fontsize=15, fontname='Times New Roman')
plt.ylabel('Standard Deviations', fontsize=15, fontname='Times New Roman')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.savefig('Plots/Returns_Over_Time.png', bbox_inches='tight')
#plt.show()

plt.figure(figsize=(11, 6))
plt.plot(datetime_objs, trading_days_data['Close'], label='RYAAY Close Price', color='#2980b9', linewidth=1)
title = "RYAAY"
plt.text(datetime_objs[200], plt.ylim()[1] * 0.85, title, fontsize=56, ha='left', va='top', fontname='Times New Roman')
plt.text(datetime_objs[250], plt.ylim()[1] * 0.70, 'Ryanair Holdings PLC', fontsize=20, ha='left', va='top', fontname='Times New Roman')
#plt.title('RYAAY Close Price Over Time', fontsize=16, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(loc='upper left', prop={'family': 'serif', 'size': 17})
plt.xlabel('Date', fontsize=15, fontname='Times New Roman')
plt.ylabel('US Dollars', fontsize=15, fontname='Times New Roman')
#plt.savefig('Plots/Close_Over_Time_with_title.png', bbox_inches='tight')
#plt.show()
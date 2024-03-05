import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('T_Ratios.csv')
df = df[252:]
df.set_index('obs', inplace=True)

dates = df.index.tolist()
datetime_objs = [datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d') for date_str in dates]

pos_sentiment_significance = []
for value in df['pos_t_ratios']:
    if value > 2.576:
        pos_sentiment_significance.append(3)
    elif value > 1.96:
        pos_sentiment_significance.append(2)
    elif value > 1.645:
        pos_sentiment_significance.append(1)
    else: pos_sentiment_significance.append(0)
    
neg_sentiment_significance = []
for value in df['neg_t_ratios']:
    if value > 2.576:
        neg_sentiment_significance.append(3)
    elif value > 1.96:
        neg_sentiment_significance.append(2)
    elif value > 1.645:
        neg_sentiment_significance.append(1)
    else: neg_sentiment_significance.append(0)

media_vol_significance = []
for value in df['med_vol_t_ratios']:
    if value > 2.576:
        media_vol_significance.append(3)
    elif value > 1.96:
        media_vol_significance.append(2)
    elif value > 1.645:
        media_vol_significance.append(1)
    else: media_vol_significance.append(0)


# Plot data
plt.figure(figsize=(12, 6))
plt.plot(datetime_objs, pos_sentiment_significance, label='Positive Sentiment Statistical Significance', color='#2980b9', linewidth=1)
plt.title('Statistical Significance of Positive Sentiment Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.xlabel('Date', fontsize=12)
plt.ylabel('Statistical Significance Level', fontsize=12)
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Plots/Positive_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()

# Plot data
plt.figure(figsize=(12, 6))
plt.plot(datetime_objs, neg_sentiment_significance, label='Negative Sentiment Statistical Significance', color='#e74c3c', linewidth=1)
plt.title('Statistical Significance of Negative Sentiment Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.xlabel('Date', fontsize=12)
plt.ylabel('Statistical Significance Level', fontsize=12)
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Plots/Negative_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()

# Plot data
plt.figure(figsize=(12, 6))
plt.plot(datetime_objs, media_vol_significance, label='Article Count Statistical Significance', color='#27ae60', linewidth=1)
plt.title('Statistical Significance of Article Count Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.xlabel('Date', fontsize=12)
plt.ylabel('Statistical Significance Level', fontsize=12)
plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Plots/Article_Count_Significance.png', bbox_inches='tight')
#plt.show()
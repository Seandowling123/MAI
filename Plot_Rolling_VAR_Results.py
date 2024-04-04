import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Rolling_VAR_Results/Returns_Rolling_VAR_P_Values.csv')
df = df[252:]
df.set_index('obs', inplace=True)

# Convert dates to objects
dates = df.index.tolist()
datetime_objs = [datetime.strptime(date_str.split(' ')[0], '%Y-%m-%d') for date_str in dates]

# Plot negative sentiment significance
fig, ax = plt.subplots(figsize=(15, 2))

# Plot each line plot on the same axis
ax.plot(datetime_objs, df['neg_p_values_lag_one'], label='Negative Sentiment Lag-1 Statistical Significance', color='#2980b9', linewidth=1)
ax.plot(datetime_objs, df['neg_p_values_lag_two'], label='Negative Sentiment Lag-2 Statistical Significance', color='#2980b9', linewidth=1)
ax.plot(datetime_objs, df['neg_p_values_lag_three'], label='Negative Sentiment Lag-3 Statistical Significance', color='#2980b9', linewidth=1)
ax.plot(datetime_objs, df['neg_p_values_lag_four'], label='Negative Sentiment Lag-4 Statistical Significance', color='#2980b9', linewidth=1)
ax.plot(datetime_objs, df['neg_p_values_lag_five'], label='Negative Sentiment Lag-5  Statistical Significance', color='#2980b9', linewidth=1)

ax.set_title('Statistical Significance of Negative Sentiment at Different Lags on Returns Over Time', fontsize=14, fontfamily='serif')
ax.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
ax.xlabel('Date', fontsize=11, fontname='Times New Roman')
ax.ylabel('Statistical Significance Level', fontsize=11, fontname='Times New Roman')
ax.yticks([1, 2, 3], ['10%', '5%', '1%'])
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.savefig('Plots/Returns_Negative_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()

# Plot data
plt.figure(figsize=(15, 2))
plt.plot(datetime_objs, media_vol_significance, label='Article Count Statistical Significance', color='#27ae60', linewidth=1)
plt.title('Statistical Significance of Lag-1 Article Count on Returns Over Time', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
plt.xlabel('Date', fontsize=11, fontname='Times New Roman')
plt.ylabel('Statistical Significance Level', fontsize=11, fontname='Times New Roman')
plt.yticks([1, 2, 3], ['10%', '5%', '1%'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Plots/Returns_Article_Count_Significance.png', bbox_inches='tight')
#plt.show()
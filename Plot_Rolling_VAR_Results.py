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

# Calculate the negative sentiment significance levels 
neg_significance_levels = [[] for _ in range(5)]
for i, p_value_series in enumerate((df['neg_p_values_lag_one'], df['neg_p_values_lag_two'], df['neg_p_values_lag_three'], df['neg_p_values_lag_four'],df['neg_p_values_lag_five'])):
    for p_value in p_value_series:
        if p_value <= 0.01:
            neg_significance_levels[i].append(3)
        elif p_value <= 0.05:
            neg_significance_levels[i].append(2)
        elif p_value <= 0.1:
            neg_significance_levels[i].append(1)
        else:
            neg_significance_levels[i].append(0)

# Plot negative sentiment significance
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 14), sharex=False, sharey=True)

# Plot each line plot on the same axis
ax1.plot(datetime_objs, neg_significance_levels[0], label='Negative Sentiment Lag-1 Statistical Significance', color='#2980b9', linewidth=1)
ax2.plot(datetime_objs, neg_significance_levels[1], label='Negative Sentiment Lag-2 Statistical Significance', color='#2980b9', linewidth=1)
ax3.plot(datetime_objs, neg_significance_levels[2], label='Negative Sentiment Lag-3 Statistical Significance', color='#2980b9', linewidth=1)
ax4.plot(datetime_objs, neg_significance_levels[3], label='Negative Sentiment Lag-4 Statistical Significance', color='#2980b9', linewidth=1)
ax5.plot(datetime_objs, neg_significance_levels[4], label='Negative Sentiment Lag-5  Statistical Significance', color='#2980b9', linewidth=1)

# Plot settings
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.set_title(f'Statistical Significance of Negative Sentiment at Lag-{i+1} on Returns Over Time', fontsize=14, fontfamily='serif')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
    ax.set_xlabel('Date', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel('Statistical Significance Level', fontsize=11, fontname='Times New Roman')
    ax.set_yticks([1, 2, 3], ['10%', '5%', '1%'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

plt.subplots_adjust(hspace=.5)
plt.savefig('Plots/Returns_Negative_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()


# Calculate the article count significance levels 
count_significance_levels = [[] for _ in range(5)]
for i, p_value_series in enumerate((df['count_p_values_lag_one'], df['count_p_values_lag_two'], df['count_p_values_lag_three'], df['count_p_values_lag_four'],df['count_p_values_lag_five'])):
    for p_value in p_value_series:
        if p_value <= 0.01:
            count_significance_levels[i].append(3)
        elif p_value <= 0.05:
            count_significance_levels[i].append(2)
        elif p_value <= 0.1:
            count_significance_levels[i].append(1)
        else:
            count_significance_levels[i].append(0)

# Plot article count significance
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 14), sharex=False, sharey=True)

# Plot each line plot on the same axis
ax1.plot(datetime_objs, count_significance_levels[0], label='Article Count Lag-1 Statistical Significance', color='#2980b9', linewidth=1)
ax2.plot(datetime_objs, count_significance_levels[1], label='Article Count Lag-2 Statistical Significance', color='#2980b9', linewidth=1)
ax3.plot(datetime_objs, count_significance_levels[2], label='Article Count Lag-3 Statistical Significance', color='#2980b9', linewidth=1)
ax4.plot(datetime_objs, count_significance_levels[3], label='Article Count Lag-4 Statistical Significance', color='#2980b9', linewidth=1)
ax5.plot(datetime_objs, count_significance_levels[4], label='Article Count Lag-5  Statistical Significance', color='#2980b9', linewidth=1)

# Plot settings
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.set_title(f'Statistical Significance of Article Count at Lag-{i+1} on Returns Over Time', fontsize=14, fontfamily='serif')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=11, loc='upper left', prop={'family': 'serif', 'size': 11})
    ax.set_xlabel('Date', fontsize=11, fontname='Times New Roman')
    ax.set_ylabel('Statistical Significance Level', fontsize=11, fontname='Times New Roman')
    ax.set_yticks([1, 2, 3], ['10%', '5%', '1%'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=10)

plt.subplots_adjust(hspace=.5)
plt.savefig('Plots/Returns_Article_Count_Significance.png', bbox_inches='tight')
#plt.show()
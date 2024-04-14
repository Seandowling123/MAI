"""
Title: Plot Rolling VAR Results
Author: Sean Dowling
Date: 15/04/2024

Description:
This script creates plots to show the statistical significance of negative sentiment on returns and absolute returns in the rolling VAR models.
It takes the p-values of the rolling VAR models coefficients and classifies them as significant at the 10%, 5% and 1% confidence levels or insignificant.
The significance of these coefficients is then plotted.

Inputs:
- P-values for the statistical significance of negative sentiment on returns (stored as Rolling_VAR_Results/Returns_Rolling_VAR_P_Values.csv)
- P-values for the statistical significance of negative sentiment on absolute returns (stored as Rolling_VAR_Results/Absolute_Returns_Rolling_VAR_P_Values.csv)

Outputs:
- Plots showing the changing statisitcal significance of negative sentiment on returns and absolute 
    returns (stored as Plots/Returns_Negative_Sentiment_Significance.png and Plots/Absolute_Returns_Negative_Sentiment_Significance.png) 

Dependencies:
- numpy (imported as np)
- pandas (imported as pd)
- datetime from the datetime module
- matplotlib.dates (imported as mdates)
- matplotlib.pyplot (imported as plt)
- Counter from the collections module

"""

import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from collections import Counter

#################################
# Plot significance for Returns #
#################################

# Load data
df = pd.read_csv('Rolling_VAR_Results/Returns_Rolling_VAR_P_Values.csv')
df = df[252:]
df.set_index('obs', inplace=True)
confidence_levels = {0: 'Insignificant', 1: '10%', 2: '5%', 3: '1%'}

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
            
# Print portion of significant models
print('Modelling Returns')
for i, significance_series in enumerate(neg_significance_levels):
    counts = Counter(significance_series)
    total_count = len(significance_series)
    percentage_counts = {num: count / total_count * 100 for num, count in counts.items()}
    print(f'Portion of Siginificant Models for Negative Sentiment Lag-{i+1}:')
    for num in confidence_levels.keys():
        print(f"{confidence_levels[num]}: {percentage_counts[num]:.2f}%")

# Plot negative sentiment significance
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 16), sharex=False, sharey=True)

# Plot each line plot on the same axis
ax1.plot(datetime_objs, neg_significance_levels[0], label='Negative Sentiment Lag-1 Statistical Significance', color='#2980b9', linewidth=1)
ax2.plot(datetime_objs, neg_significance_levels[1], label='Negative Sentiment Lag-2 Statistical Significance', color='#2980b9', linewidth=1)
ax3.plot(datetime_objs, neg_significance_levels[2], label='Negative Sentiment Lag-3 Statistical Significance', color='#2980b9', linewidth=1)
ax4.plot(datetime_objs, neg_significance_levels[3], label='Negative Sentiment Lag-4 Statistical Significance', color='#2980b9', linewidth=1)
ax5.plot(datetime_objs, neg_significance_levels[4], label='Negative Sentiment Lag-5 Statistical Significance', color='#2980b9', linewidth=1)

# Plot settings
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.set_title(f'Statistical Significance of Negative Sentiment at Lag-{i+1} on Returns Over Time', fontsize=18, fontfamily='serif')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=13, loc='upper left', prop={'family': 'serif', 'size': 13})
    ax.set_xlabel('Date', fontsize=15, fontname='Times New Roman')
    ax.set_ylabel('Significance Level', fontsize=15, fontname='Times New Roman')
    ax.set_yticks([1, 2, 3], ['10%', '5%', '1%'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=13)

plt.subplots_adjust(hspace=.6)
plt.savefig('Plots/Returns_Negative_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()


##########################################
# Plot significance for Absolute Returns #
##########################################

# Load data
df = pd.read_csv('Rolling_VAR_Results/Absolute_Returns_Rolling_VAR_P_Values.csv')
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
            
# Print portion of significant models
print('\nModelling Absolute Returns')
for i, significance_series in enumerate(neg_significance_levels):
    counts = Counter(significance_series)
    total_count = len(significance_series)
    percentage_counts = {num: count / total_count * 100 for num, count in counts.items()}
    print(f'Portion of Siginificant Models for Negative Sentiment Lag-{i+1}:')
    for num in confidence_levels.keys():
        print(f"{confidence_levels[num]}: {percentage_counts[num]:.2f}%")

# Plot negative sentiment significance
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(16, 16), sharex=False, sharey=True)

# Plot each line plot on the same axis
ax1.plot(datetime_objs, neg_significance_levels[0], label='Negative Sentiment Lag-1 Statistical Significance', color='#2980b9', linewidth=1)
ax2.plot(datetime_objs, neg_significance_levels[1], label='Negative Sentiment Lag-2 Statistical Significance', color='#2980b9', linewidth=1)
ax3.plot(datetime_objs, neg_significance_levels[2], label='Negative Sentiment Lag-3 Statistical Significance', color='#2980b9', linewidth=1)
ax4.plot(datetime_objs, neg_significance_levels[3], label='Negative Sentiment Lag-4 Statistical Significance', color='#2980b9', linewidth=1)
ax5.plot(datetime_objs, neg_significance_levels[4], label='Negative Sentiment Lag-5 Statistical Significance', color='#2980b9', linewidth=1)

# Plot settings
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.set_title(f'Statistical Significance of Negative Sentiment at Lag-{i+1} on Absolute Returns Over Time', fontsize=18, fontfamily='serif')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(fontsize=13, loc='upper left', prop={'family': 'serif', 'size': 13})
    ax.set_xlabel('Date', fontsize=15, fontname='Times New Roman')
    ax.set_ylabel('Significance Level', fontsize=15, fontname='Times New Roman')
    ax.set_yticks([1, 2, 3], ['10%', '5%', '1%'])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=13)

plt.subplots_adjust(hspace=.6)
plt.savefig('Plots/Absolute_Returns_Negative_Sentiment_Significance.png', bbox_inches='tight')
#plt.show()
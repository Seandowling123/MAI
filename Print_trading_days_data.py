import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('T_Ratios.csv')
df.set_index('obs', inplace=True)

# Select relevant columns for modeling
df = df[['pos_t_ratios', 'med_vol_t_ratios']]

pos_sentiment_significance = []
for value in df['pos_t_ratios']:
    if value > 2.576:
        pos_sentiment_significance.append(3)
    elif value > 1.96:
        pos_sentiment_significance.append(2)
    elif value > 1.645:
        pos_sentiment_significance.append(1)
    else: pos_sentiment_significance.append(0)

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
plt.figure(figsize=(10, 6))
plt.plot(df.index, pos_sentiment_significance, label='T-ratios Positive Sentiment', color='blue')
plt.plot(df.index, media_vol_significance, label='T-ratios Media Volume', color='#e74c3c')
plt.title('VAR Forecasting with Training and Testing Data')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()
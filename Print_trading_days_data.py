import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('XX_daily_data.csv')
df.set_index('Date', inplace=True)

# Select relevant columns for modeling
df = df[['Returns', 'Positive_Sentiment']]

# Plot data
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Returns'], label='Actual Returns')
plt.plot(df.index, df['Positive_Sentiment'], label='Sentiment')
plt.title('VAR Forecasting with Training and Testing Data')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.show()
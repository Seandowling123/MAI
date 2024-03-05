import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd
import numpy as np

def get_coefficients(file_path, lag_length):
    with open(file_path, 'r', encoding='utf-8') as file:
        numbers_list = []
        variable_names = []
        
        # Get variables
        for line in file:
            line = line.replace('−', '-').replace('*', '')
            first_space_index = line.find(' ')
            variable_names.append(line[:first_space_index].strip().rsplit('_', 1)[0])
            numbers = [float(f'{float(num):.6f}') for num in line[first_space_index:].split() if num.strip()]
            numbers_list.append(numbers[0])
        
        coefficients = []
        prev_variable = ''
        variable_num = -1
        
        # Get the coefficients for each variable
        for i in range(len(numbers_list)):
            if variable_names[i] != prev_variable:
                coefficients.append([])
                variable_num = variable_num+1
            coefficients[variable_num].append(numbers_list[i])
            prev_variable = variable_names[i]
        return coefficients

def get_trading_days_data(file_path):
    df = pd.read_csv(file_path)
    df.set_index('Date', inplace=True)
    return df

def get_VAR_estimation(trading_days_data, coefficients, lag_length):
    VAR_estimations = []
    
    # Get extimation for each day
    for i in range(lag_length+1,len(trading_days_data)):
        
        # Const
        const = 1*coefficients[0][0]
        
        # Get lagged endogenous variables
        lagged_returns = (list(trading_days_data['Returns'][i-lag_length-1:i-1]))[::-1]
        lagged_pos_sent = (list(trading_days_data['Stemmed_Positive_Sentiment'][i-lag_length-1:i-1]))[::-1]
        lagged_neg_sent = (list(trading_days_data['Stemmed_Negative_Sentiment'][i-lag_length-1:i-1]))[::-1]
        lagged_media_vol = (list(trading_days_data['Media_Volume'][i-lag_length-1:i-1]))[::-1]
        lagged_VIX = (list(trading_days_data['VIX_Close'][i-lag_length-1:i-1]))[::-1]
        lagged_volume = (list(trading_days_data['Detrended_Volume'][i-lag_length-1:i-1]))[::-1]

        # Multiply by weights
        weighted_returns = np.dot(lagged_returns, coefficients[1])
        weighted_pos_sent = np.dot(lagged_pos_sent, coefficients[2])
        weighted_neg_sent = np.dot(lagged_neg_sent, coefficients[3])
        weighted_media_vol = np.dot(lagged_media_vol, coefficients[4])
        weighted_VIX = np.dot(lagged_VIX, coefficients[5])
        weighted_volume = np.dot(lagged_volume, coefficients[6])
        
        # Exogenous variables
        weighted_monday = trading_days_data['Monday'][i] * coefficients[7][0]
        weighted_january = trading_days_data['January'][i] * coefficients[8][0]
        weighted_crash = trading_days_data['Crash'][i] * coefficients[9][0]
        
        # Get VAR returns estimation for that day
        VAR_estimations.append(np.sum([const,weighted_returns,weighted_pos_sent,weighted_neg_sent,weighted_media_vol,weighted_VIX,weighted_volume,weighted_monday,weighted_january,weighted_crash]))
    return VAR_estimations
    
def trading_strat(trading_days_data, VAR_estimations):
    buy = False
    trading_returns = []
    trading_profit = []
    prev_close = 0
    current_profit = 0
    
    for i in range(len(VAR_estimations)):
        
        # Calculate trading decision
        if VAR_estimations[i] > 0:
            buy = True
        else: buy = False
        
        # Evaluate returns
        if buy:
            current_profit = (trading_days_data['Close'][i] - prev_close) + current_profit
            trading_profit.append(current_profit)
            trading_returns.append(trading_days_data['Returns'][i])
        else:
            trading_profit.append(current_profit)
            trading_returns.append(0)
        prev_close = trading_days_data['Close'][i]
    return trading_returns, trading_profit
        

lag_length = 5
VAR_file_name = 'VAR_Results.txt'
trading_days_file_name = 'XX_daily_data.csv'
coefficients = get_coefficients(VAR_file_name, lag_length)
trading_days_data = get_trading_days_data(trading_days_file_name)
print(trading_days_data.head())
VAR_estimations = get_VAR_estimation(trading_days_data, coefficients, lag_length)
trading_returns, trading_profit = trading_strat(trading_days_data[lag_length+1:], VAR_estimations)
print(f"Average profits: VAR: {np.mean(trading_profit)} | Normal: {np.mean(list(trading_days_data['Close'][lag_length+1:]))}")

"""# Plot data
plt.figure(figsize=(10, 6))
plt.plot(trading_days_data.index[lag_length+1:], trading_days_data['Returns'][lag_length+1:], label='Returns', color='blue')
plt.plot(trading_days_data.index[lag_length+1:], VAR_estimations, label='VAR Estimation', color='#e74c3c')
plt.plot(trading_days_data.index[lag_length+1:], trading_returns, label='VAR Estimation', color='green')
plt.title('VAR Forecasting with Training and Testing Data')
plt.xticks(range(0, len(trading_days_data.index[lag_length+1:]), 500))
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
#plt.show()"""

dates = trading_days_data.index[lag_length+1:].tolist()
datetime_objs = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in dates]

# Plot data
plt.figure(figsize=(12, 6))
plt.plot(datetime_objs, trading_days_data['Close'][lag_length+1:], label='Buy-and-Hold Strategy', color='#2980b9', linewidth=1)
plt.plot(datetime_objs, trading_profit, label='VAR Using Sentiment Strategy', color='#e74c3c', linewidth=1)
plt.title('Trading Profits Using VAR with Sentiment Vs Buy-and-Hold Strategy', fontsize=14, fontfamily='serif')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.legend(fontsize=11, loc='upper left')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Profit (US Dollars)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig('Plots/Trading_Strategy_Profits.png', bbox_inches='tight')
plt.show()




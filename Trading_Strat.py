import matplotlib.pyplot as plt
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
            variable_names.append(line[:first_space_index].strip().split('_')[0])
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
    for i in range(6,len(trading_days_data)):
        # Const
        const = 1*coefficients[0][0]
        
        # Get lagged endogenous variables
        lagged_returns = (list(trading_days_data['Returns'][i-lag_length-1:i-1]))[::-1]
        lagged_pos_sent = (list(trading_days_data['Positive_Sentiment'][i-lag_length-1:i-1]))[::-1]
        lagged_media_vol = (list(trading_days_data['Media_Volume'][i-lag_length-1:i-1]))[::-1]
        lagged_VIX = (list(trading_days_data['VIX_Close'][i-lag_length-1:i-1]))[::-1]

        # Multiply by weights
        weighted_returns = np.dot(lagged_returns, coefficients[1])
        weighted_pos_sent = np.dot(lagged_pos_sent, coefficients[2])
        weighted_media_vol = np.dot(lagged_media_vol, coefficients[3])
        weighted_VIX = np.dot(lagged_VIX, coefficients[4])
        
        # Exogenous variables
        weighted_monday = trading_days_data['Monday'][i] * coefficients[5][0]
        
        # Get VAR returns estimation for that day
        VAR_estimations.append(np.sum([const, weighted_returns, weighted_pos_sent, weighted_media_vol, weighted_VIX, weighted_monday]))
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
trading_returns, trading_profit = trading_strat(trading_days_data[6:], VAR_estimations)

# Plot data
plt.figure(figsize=(10, 6))
plt.plot(trading_days_data.index[6:], trading_days_data['Returns'][6:], label='Returns', color='blue')
plt.plot(trading_days_data.index[6:], VAR_estimations, label='VAR Estimation', color='#e74c3c')
plt.plot(trading_days_data.index[6:], trading_returns, label='VAR Estimation', color='green')
plt.title('VAR Forecasting with Training and Testing Data')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
#plt.show()

# Plot data
plt.figure(figsize=(10, 6))
plt.plot(trading_days_data.index[6:], trading_days_data['Close'][6:], label='Close', color='blue')
plt.plot(trading_days_data.index[6:], trading_profit, label='Profit', color='green')
plt.title('VAR Forecasting with Training and Testing Data')
plt.xlabel('Date')
plt.ylabel('Close')
plt.legend()
plt.show()




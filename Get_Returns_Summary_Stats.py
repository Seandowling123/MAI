"""
Title: Get Returns Summary Stats
Author: Sean Dowling
Date: 15/04/2024

Description:
This script splits the 20-year investigation period into 10 2-year intervals and prints the summary statistics for returns during each one

Usage:
The script can be run without interaction, povided that the relevant inputs are available.

Inputs:
- Historical stock data (stored as Financial_Data/RYAAY.csv)

Outputs:
- Summary statistics for each 2-year interval are printed in the terminal

Dependencies:
- csv
- datetime from the datetime module
- pandas (imported as pd)
- math
- numpy (imported as np)
- mode and median from the statistics module
- variance from the statistics module
- skew and kurtosis from the scipy.stats module

"""

import csv
from datetime import datetime
import pandas as pd
import math
import numpy as np
from statistics import mode, median, variance
from scipy.stats import skew, kurtosis

# Get the indicies to split the time series into 2-year intervals
def get_split_indices(dates):
    # Initialize variables
    split_indices = []
    current_year = dates[0].year
    start_index = 0

    # Iterate over the dates list
    for i, date in enumerate(dates):
        if date.year != current_year:
            split_indices.append((start_index, i - 1))
            current_year = date.year
            start_index = i

    # Add the last 2-year period
    split_indices.append((start_index, len(dates) - 1))
    return split_indices

def get_descriptive_stats(returns):
    
    # Central tendancy
    mean_returns = np.mean(returns)
    mode_returns = mode(returns)
    median_returns = median(returns)
    
    # Spread
    dev_returns = np.std(returns)
    sample_var_return = variance(returns)
    data_range_return = max(returns) - min(returns)
    
    # Skewness / Kurtosis
    data_skewness = skew(returns)
    data_kurtosis = kurtosis(returns)
    
    # Autocorrelation
    lag = 5
    lags = range(1, lag + 1)
    df = pd.DataFrame(returns, columns=['Returns'])
    autocorrelations = [df['Returns'].autocorr(lag=lag) for lag in lags]
    
    # Min / Max
    min_returns = min(returns)
    max_returns = max(returns)
    
    # Print the stats
    print("\nRYAAY Descriptive Statistics")
    print("Mean & {:.4f}".format(mean_returns))
    print("Mode & {:.4f}".format(mode_returns))
    print("Median & {:.4f}".format(median_returns))
    print("Standard Deviation & {:.4f}".format(dev_returns))
    print("Sample Variance & {:.4f}".format(sample_var_return))
    print("Range & {:.4f}".format(data_range_return))
    print("Skewness & {:.4f}".format(data_skewness))
    print("Kurtosis & {:.4f}".format(data_kurtosis))
    for lag, autocorr in zip(range(1, lag + 1), autocorrelations):
        print("Autocorrelation at Lag", lag, "& {:.4f}".format(autocorr))
    print("Minimum & {:.4f}".format(min_returns))
    print("Maximum & {:.4f}".format(max_returns))
    print("Data Points & {:.4f}".format(len(returns)))

# Get the historical stock closing prices
def get_close_prices(input_file_path, start_date, end_date):
    prices = []
    dates = []
    try:
        with open(input_file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)

            for row in reader:
                date_str = row['Date']
                date_object = datetime.strptime(date_str, '%Y-%m-%d')

                if start_date <= date_object <= end_date:
                    close_price = float(row['Adj Close'])
                    prices.append(close_price)
                    dates.append(date_object)

        return prices, dates
    except FileNotFoundError:
        print(f"File not found: {input_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None
    
# Extract data
start_date = "2004-01-01"
end_date = "2023-12-31"
close_prices, dates = get_close_prices("Financial_Data/RYAAY.csv", datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d'))
returns = []
for i in range(len(close_prices)):
    if i != 0:
        returns.append(math.log(close_prices[i]/close_prices[i-1]))

# Split the period into 2-year intervals and get summary statistics of each
split_indices = get_split_indices(dates)
for i in range(10):
    start = split_indices[i*2][0]
    end = split_indices[i*2+1][1]
    print(f'\nPeriod: {dates[start]} - {dates[end]}', end='', flush=True)
    get_descriptive_stats(returns[start:end])
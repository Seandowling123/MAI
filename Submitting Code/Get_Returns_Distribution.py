"""
Title: Get Returns Distribution
Author: Sean Dowling
Date: 15/04/2024

Description:
This script analyses the distribution of returns over the investigation period. This distribution is also compared to the normal distribution.

Inputs:
- Historical stock data (stored as Financial_Data/RYAAY.csv)

Outputs:
- A comparison of the distribution or returns and the normal distribution is printed in the terminal

Dependencies:
- csv
- datetime from the datetime module
- math
- numpy (imported as np)
- norm from the scipy.stats module

"""

import csv
from datetime import datetime
import math
import numpy as np
from scipy.stats import norm

# Calculate the distribution of the returns time series & compare with normal distribution
def get_distribution_data(returns):
    
    # Calculate mean and standard deviation of returns
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)
    
    deviations = [.25, .5, .75, 1, 1.5, 2, 2.5 ,3, 4, 5, 6]
    returns_cdf = {}
    normal_cdf = {}
    
    # Get normal distribution stats
    for deviation in deviations:
        normal_cdf[deviation] = (norm.cdf(deviation, 0, 1) - norm.cdf(-deviation, 0, 1))*100
        
    # Get returns distribution stats
    for daily_return in returns:
        z_score = (daily_return-mean_returns)/std_returns
        for deviation in deviations:
            if np.abs(z_score) < deviation:
                (norm.cdf(-deviation, 1, 1) - norm.cdf(deviation, 1, 1))*100
                if deviation in returns_cdf:
                    returns_cdf[deviation] = returns_cdf[deviation]+1
                else: returns_cdf[deviation] = 0
    
    for deviation in returns_cdf:
        returns_cdf[deviation] = (returns_cdf[deviation] / len(returns))*100
    
    # Print return stats
    print("\nCDF Data:")
    prev_deviation = 0
    returns_cdf_prev = 0
    norm_cdf_prev = 0
    for deviation in deviations:
        returns_cdf_delta = returns_cdf[deviation]-returns_cdf_prev
        norm_cdf_delta = normal_cdf[deviation]-norm_cdf_prev
        print(f"textbf{{Std. Deviations: {float(prev_deviation):.2f} - {float(deviation):.2f}}} | RYAAY: {returns_cdf_delta:.2f} | Normal: {norm_cdf_delta:.2f} | Disc.: {(returns_cdf_delta-norm_cdf_delta):.2f}")
        prev_deviation = deviation
        returns_cdf_prev = returns_cdf[deviation]
        norm_cdf_prev = normal_cdf[deviation]
    print(f"textbf{{Std. Deviations: > 6}}         | RYAAY: {100-returns_cdf[6]:.2f} | Normal: {100-normal_cdf[6]:.2f} | Disc.: {(100-returns_cdf[6]-(100-normal_cdf[6])):.2f}")

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

# Get dates for financial crashes
def get_crash_dates_intervals():
    gfc_start_date = datetime(2007, 12, 1)
    gfc_end_date = datetime(2009, 6, 30)
    covid_start_date = datetime(2020, 2, 1)
    covid_end_date = datetime(2020, 4, 30)
    
    return [(gfc_start_date, gfc_end_date), (covid_start_date, covid_end_date)]

# Extract data
start_date = "2004-01-01"
end_date = "2023-12-31"
close_prices, dates = get_close_prices("Financial_Data/RYAAY.csv", datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d'))
returns = []
for i in range(len(close_prices)):
    if i != 0:
        returns.append(math.log(close_prices[i]/close_prices[i-1]))

# Calculate mean and standard deviation of returns
mean_returns = np.mean(returns)
std_returns = np.std(returns)

# Print the returns returns stats
get_distribution_data(returns)
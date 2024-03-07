import csv
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt
import math
import numpy as np
from statistics import mode, median, variance
from scipy.stats import norm, skew, kurtosis, jarque_bera

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
    
    # Get return stats
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
    
    print("\nCDF Data (cum):")
    for deviation in deviations:
        print(f"Std. Deviation: {float(deviation):.2f} | RYAAY: {returns_cdf[deviation]:.2f} | Normal: {normal_cdf[deviation]:.2f} | Disc.: {(returns_cdf[deviation]-normal_cdf[deviation]):.2f}")
    
    print("\nCDF Data: (range)")
    prev_deviation = 0
    returns_cdf_prev = 0
    norm_cdf_prev = 0
    for deviation in deviations:
        returns_cdf_delta = returns_cdf[deviation]-returns_cdf_prev
        norm_cdf_delta = normal_cdf[deviation]-norm_cdf_prev
        print(f"textbf{{Std. Deviation: {float(prev_deviation):.2f} - {float(deviation):.2f}}} & {returns_cdf_delta:.2f} & {norm_cdf_delta:.2f} & {(returns_cdf_delta-norm_cdf_delta):.2f} \\\\")
        prev_deviation = deviation
        returns_cdf_prev = returns_cdf[deviation]
        norm_cdf_prev = normal_cdf[deviation]
    print(f"textbf{{Std. Deviation: > 6}}         & {100-returns_cdf[6]:.2f} & {100-normal_cdf[6]:.2f} & {(100-returns_cdf[6]-(100-normal_cdf[6])):.2f} \\\\")

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
    lag = 10
    lags = range(1, lag + 1)
    autocorrelations = [np.corrcoef(returns[:-lag], returns[lag:])[0, 1] for lag in lags]
    
    # Min / Max
    min_returns = min(returns)
    max_returns = max(returns)
    
    # Jarque-Bera
    test_statistic, p_value = jarque_bera(returns)
    
    # Print the stats
    print("\nRYAAY Descriptive Statistics", "\\\\")
    print("textbf{Mean} & ", mean_returns, "\\\\")
    print("textbf{Mode} & ", mode_returns, "\\\\")
    print("textbf{Median} & ", median_returns, "\\\\")
    print("textbf{Standard Deviation} & ", dev_returns, "\\\\")
    print("textbf{Sample Variance} & ", sample_var_return, "\\\\")
    print("textbf{Range} & ", data_range_return, "\\\\")
    print("textbf{Skewness} & ", data_skewness, "\\\\")
    print("textbf{Kurtosis} & ", data_kurtosis, "\\\\")
    for lag, autocorr in zip(range(1, lag + 1), autocorrelations):
        print("textbf{Autocorrelation at Lag", lag, "&", autocorr, "\\\\")
    print("textbf{Minimum} & ", min_returns, "\\\\")
    print("textbf{Maximum} & ", max_returns, "\\\\")
    print("textbf{Count} & ", len(returns), "\\\\")
    print("textbf{Jarque-Bera p-value} & ", p_value, "\\\\")

def ectract_close_prices(input_file_path, start_date, end_date):
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
start_date = "2003-01-01"
end_date = "2023-12-31"
close_prices, dates = ectract_close_prices("RYAAY.csv", datetime.strptime(start_date, '%Y-%m-%d'), datetime.strptime(end_date, '%Y-%m-%d'))
returns = []
for i in range(len(close_prices)):
    if i != 0:
        returns.append(math.log(close_prices[i]/close_prices[i-1]))

# Calculate mean and standard deviation of returns
mean_returns = np.mean(returns)
std_returns = np.std(returns)

# Print the returns returns stats
get_distribution_data(returns)
get_descriptive_stats(returns)
get_descriptive_stats(np.abs(returns))

###################
# Show Distribution
###################

# Calculate outliers
num_outliers = 0
returns_zscores = []

# Remove outliers
for i in range(len(returns)):
    if np.abs(returns[i]) < 4*std_returns:
        returns_zscores.append((returns[i]-mean_returns)/std_returns)

# Create a range of x values
x_values = np.linspace(-4, 4, 100)
# Overlay normal distribution curve
normal_distribution = norm.pdf(x_values, mean_returns, 1)
# Create histogram
num_bins = 50
plt.figure(figsize=(8.5, 6))
plt.hist(returns_zscores, bins=num_bins, color='#2980b9', alpha=0.7, density=True, label='Daily RYAAY Returns Distribution', edgecolor='black', linewidth=0.5)
plt.plot(x_values, normal_distribution, color='#e74c3c', label='Normal Distribution', linewidth=1)
# Show histogram
plt.fill_between(x_values, normal_distribution, alpha=0.2, color='#e74c3c')
plt.xlabel('Returns', fontsize=12)
plt.ylabel('Probability Density', fontsize=13, fontname='Times New Roman')
plt.xlabel('Standard Deviations', fontsize=13, fontname='Times New Roman')
plt.title('Distribution of RYAAY Returns and Normal Distribution', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10, prop={'family': 'serif', 'size': 10})
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.show()
# Save the plot as a PNG file
plt.savefig('Plots/NormalDistributionRYAAYAdjustedReturns.png', bbox_inches='tight')
plt.close()

###########################
# Absolute prices over time
###########################

# Calculate 30-period moving average
ma_window = 30
moving_average = np.convolve(np.abs(returns), np.ones(ma_window)/ma_window, mode='valid')
# Create plot
plt.figure(figsize=(12, 6))
plt.plot(dates[1:], np.abs(returns), color='#2980b9', label='Daily Absolute Returns', linewidth=1)
plt.plot(dates[ma_window//2:-ma_window//2], moving_average, color='#e74c3c', label='30-day Moving Average', linewidth=1)

# Adding crash data
for start_date, end_date in get_crash_dates_intervals():
    plt.axvspan(start_date, end_date, color='lightgrey', alpha=0.9)
plt.text(get_crash_dates_intervals()[0][1], plt.ylim()[1] * 0.9, 'Global Financial Crisis', horizontalalignment='center', fontname='Times New Roman', fontsize=11)
plt.text(get_crash_dates_intervals()[1][1], plt.ylim()[1] * 0.9, 'COVID-19 Crash', horizontalalignment='center', fontname='Times New Roman', fontsize=11)

plt.xlabel('Time (Trading Days)', fontsize=13, fontname='Times New Roman')
plt.ylabel('Absolute Returns', fontsize=13, fontname='Times New Roman')
plt.title('Absolute Returns Over Time', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10, loc='upper left', prop={'family': 'serif', 'size': 10})
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.show()
# Save the plot as a PNG file
plt.savefig('Plots/absolute_returns_plot.png', bbox_inches='tight')
plt.close()


#####################
# Autocorrelation at lags
#####################

num_lags = 252
correlations = []
for i in range(1,num_lags):
    correlations.append(np.corrcoef(returns[i:], returns[:(len(returns)-i)])[0, 1])
absolute_correlations = []
for i in range(1,num_lags):
    absolute_correlations.append(np.corrcoef(np.abs(returns[i:]), np.abs(returns[:(len(returns)-i)]))[0, 1])
# Create plot
plt.figure(figsize=(12, 6))
plt.plot(range(1,num_lags), absolute_correlations, color='#2980b9', label='Absolute Returns Autocorrelation', linewidth=1)
plt.fill_between(range(1,num_lags), absolute_correlations, alpha=0.2, color='#2980b9')
plt.plot(range(1,num_lags), correlations, color='#e74c3c', label='Returns Autocorrelation', linewidth=1)
plt.fill_between(range(1,num_lags), correlations, alpha=0.2, color='#e74c3c')
plt.xlabel('Lag (Trading Days)', fontsize=13, fontname='Times New Roman')
plt.ylabel('Correlation Coefficient (R)', fontsize=13, fontname='Times New Roman')
plt.title('Autocorrelation of RYAAY Returns & Absolute Returns At Different Time Lags', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10, prop={'family': 'serif', 'size': 10})
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
# Save the plot as a PNG file
plt.savefig('Plots/absolute_returns_correlation_plot.png', bbox_inches='tight')
#plt.show()
plt.close()
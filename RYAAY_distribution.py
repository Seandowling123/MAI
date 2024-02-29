import csv
from datetime import datetime
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Tahoma']
import matplotlib.pyplot as plt
import math
import numpy as np
import time
from scipy.stats import norm

def get_distribution_data(returns):
    
    # Calculate mean and standard deviation of returns
    mean_returns = np.mean(returns)
    std_returns = np.std(returns)

    # Define Normal distribution
    #cdf_lower = norm.cdf(lower_bound * sigma, mu, sigma)
    #cdf_upper = norm.cdf(upper_bound * sigma, mu, sigma)
    #percentage = (cdf_upper - cdf_lower) * 100

    # Calculate outliers
    num_outliers = 0
    returns_zscores = []
    deviation1 = 0
    deviation2 = 0
    deviation3 = 0
    # Remove outliers
    for i in range(len(returns)):
        if np.abs(returns[i]) < 4*std_returns:
            returns_zscores.append((returns[i]-mean_returns)/std_returns)
            if np.abs(returns[i]) < 3*std_returns:
                deviation3 = deviation3+1
                if np.abs(returns[i]) < 2*std_returns:
                    deviation2 = deviation2+1
                    if np.abs(returns[i]) < 1*std_returns:
                        deviation1 = deviation1+1
            else: num_outliers = num_outliers+1
        else: num_outliers = num_outliers+1
    # Get distribution stats
    outlier_percentage = num_outliers/len(returns)
    deviation_percentage_3 = deviation3/len(returns)
    deviation_percentage_2 = deviation2/len(returns)
    deviation_percentage_1 = deviation1/len(returns)
    print("Outliers:", outlier_percentage, "| Deviations:", deviation_percentage_3, deviation_percentage_2, deviation_percentage_1)
    

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
    
# Extract returns
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

# Calculate outliers
num_outliers = 0
returns_zscores = []
deviation1 = 0
deviation2 = 0
deviation3 = 0
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
plt.hist(returns_zscores, bins=num_bins, color='#2980b9', alpha=0.7, density=True, label='Daily RYAAY Returns', edgecolor='black', linewidth=0.5)
plt.plot(x_values, normal_distribution, color='#e74c3c', label='Normal Distribution', linewidth=1)
# Show histogram
plt.fill_between(x_values, normal_distribution, alpha=0.2, color='#e74c3c')
plt.xlabel('Returns', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.xlabel('Standard Deviations', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Histogram of Normal Distribution and RYAAY Returns', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.show()
# Save the plot as a PNG file
plt.savefig('Plots/NormalDistributionRYAAYAdjustedReturns.png', bbox_inches='tight')
plt.close()

# Absolute prices over time
# Calculate 10-period moving average
ma_window = 30
moving_average = np.convolve(np.abs(returns), np.ones(ma_window)/ma_window, mode='valid')
# Create plot
plt.figure(figsize=(12, 6))
plt.plot(dates[1:], np.abs(returns), color='#2980b9', label='Daily Absolute Returns', linewidth=1)
plt.plot(dates[ma_window//2:-ma_window//2], moving_average, color='#e74c3c', label='30-day Moving Average', linewidth=1)
plt.xlabel('Time', fontsize=12)
plt.ylabel('Absolute Returns', fontsize=12)
plt.title('Absolute Returns Over Time', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
#plt.show()
# Save the plot as a PNG file
plt.savefig('Plots/absolute_returns_plot.png', bbox_inches='tight')
plt.close()

# Correlation at lags
num_lags = 252
correlations = []
for i in range(1,num_lags):
    correlations.append(np.corrcoef(returns[i:], returns[:(len(returns)-i)])[0, 1])
absolute_correlations = []
for i in range(1,num_lags):
    absolute_correlations.append(np.corrcoef(np.abs(returns[i:]), np.abs(returns[:(len(returns)-i)]))[0, 1])
# Create plot
plt.plot(range(1,num_lags), absolute_correlations, color='#2980b9', label='Absolute Returns Autocorrelation', linewidth=1)
plt.fill_between(range(1,num_lags), absolute_correlations, alpha=0.2, color='#2980b9')
plt.plot(range(1,num_lags), correlations, color='#e74c3c', label='Returns Autocorrelation', linewidth=1)
plt.fill_between(range(1,num_lags), correlations, alpha=0.2, color='#e74c3c')
plt.xlabel('Lag (Trading Days)', fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
plt.title('Autocorrelation of RYAAY Returns & Absolute Returns At Different Time Lags', fontsize=14, fontfamily='serif')
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tick_params(axis='both', which='major', labelsize=10)
# Save the plot as a PNG file
plt.savefig('Plots/absolute_returns_correlation_plot.png', bbox_inches='tight')
#plt.show()
plt.close()
import numpy as np
import pandas as pd
import math
import csv
import time
import re
import os
from datetime import datetime
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# Data to save for each trading day
class Trading_Day:
    def __init__(self, date, close, returns, absolute_returns, sentiment):
        self.date = date
        self.close = close
        self.returns = returns
        self.absolute_returns = absolute_returns
        self.sentiment = sentiment
    
    def to_csv_line(self):
        return f"{str(self.date)},{str(self.close)},{str(self.returns)},{str(self.absolute_returns)},{str(self.sentiment)}"

# Class containing info about each article
class Article:
    def __init__(self, date, body, headline, sentiment):
        self.date = date
        self.body = body
        self.headline = headline
        self.sentiment = sentiment

# Load dictionary words from csv
def load_csv(file_path):
    try:
        with open(file_path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            entries = [row[0] for row in reader]
        return entries
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Load articles from text file
def load_articles_from_txt(file_path):
    try:
        with open(file_path, 'r', encoding="latin-1") as file:
            content = file.read()
            content = content.replace('\xa0', '')
            articles = content.split('End of Document')
            del articles[-1]
        return articles
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

# Pre-process articles
def process_text(body):
    try:
        # Extract article body & filter content
        if "\nBody\n" in body:
            body_split = (((body.split("\nBody\n")[1]).split('Load-Date:')[0]).split("\nNotes\n")[0])
            body_split = (body_split.replace('\n', '')).replace('  ', '')
            body_filtered = re.sub(r'[^a-zA-Z ]', '', body_split)
            body_upper = body_filtered.upper()
            return body_upper
        else: 
            print("Error: Article body could not be found.")
            return 0, 0
    except Exception as e:
        print("Error processing text")
        return 0, 0

# Returns a dateTime object for news articles
def convert_string_to_datetime(date_string):
    try:
        datetime_object = datetime.strptime(date_string, '%B %d, %Y')
        return datetime_object
    except ValueError:
        print(f"Unable to parse the date string: {date_string}")
        return f"Unable to parse the date string: {date_string}"

# Extracts date and body of each news article
def extract_article_data(raw_articles):
    articles = []
    dates = []
    
    # Extract data
    for i in range(len(raw_articles)):
        headline = raw_articles[i].split("\n")[0]
        date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b \d{1,2}, \d{4}')
        match = date_pattern.search(raw_articles[i])
        
        # Check for valid date
        if match:
            date_string = match.group()
            date = convert_string_to_datetime(date_string)
            
            # Add to Articles list
            if isinstance(date, datetime):
                dates.append(date)
                body = process_text(raw_articles[i])
                if body != 0:
                    articles.append(Article(date, body, headline, 0))
                else: print("Removed article. Incorrect syntax")
            else: print("Removed article. Date loaded incorrectly")
        else: print("Removed article. Date not found")
    print("Loaded", len(articles), "articles")
    return articles, dates

# Count the number of dictionary words in an article
def get_word_count(article, word_list):
    word_counts = 0
    for word in word_list:
        count = article.count(word)
        word_counts = word_counts + count
    return word_counts

# Calculate sentiment score
def get_sentiment_scores(articles, positive_dict, negative_dict):
    calculated = 0
    total = len(articles)
    for article in articles:
        try:
            # Get counts
            num_words = len(word_tokenize(article.body))
            pos_word_count = get_word_count(article.body, positive_dict)
            neg_word_count = get_word_count(article.body, negative_dict)
            
            # Calculate relative word frequencies
            pos_score = pos_word_count/num_words
            neg_score = neg_word_count/num_words
            total_score = pos_score - neg_score
            
            # Save score
            article.sentiment = total_score
            
            # Progress tracker
            calculated = calculated + 1
            progress = "{:.2f}".format((calculated/total)*100)
            if (calculated % 100) == 0:
                print(f"Calculating Sentiment: {progress}%\r", end='', flush=True)
            
        except Exception as e:
            print(f"An sentiment calculation error occurred: {str(e)}")

# Extract financial data 
def ectract_close_prices(file_path, start_date, end_date):
    filtered_data_dict = {}
    try:
        with open(file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)

            for row in reader:
                date_str = row['Date']
                date_object = datetime.strptime(date_str, '%Y-%m-%d')

                if start_date <= date_object <= end_date:
                    close_price = float(row['Adj Close'])
                    filtered_data_dict[date_object] = close_price

        return filtered_data_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

# Collect data for each trading day
def get_trading_day_data(daily_senitment, close_prices):
    trading_days = {}
    prev_date = 0
    for date in close_prices:
        if prev_date != 0:
            # Calculate data
            close = close_prices[date]
            returns = math.log(close_prices[date]/close_prices[prev_date])
            if date in daily_senitment:
                senitment = daily_senitment[date]
            else: senitment = 0
            # Store in trading days dict
            trading_days[date] = Trading_Day(date, close, returns, abs(returns), senitment)
        prev_date = date
    return trading_days

def save_trading_days_to_csv(trading_days, csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Date", "Close", "Returns", "AbsoluteReturns", "Sentiment"])  # Writing header

            for date, trading_day in trading_days.items():
                writer.writerow(trading_day.to_csv_line().split(','))

        print(f"Trading days data saved to {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Select mode
mode  = "tes"

if mode == "test":
    articles_file_path = 'Sample_article.txt'
else: articles_file_path = 'Articles_txt_combined/Articles_combined.txt'

# Load files
#articles_file_path = 'Sample_article.txt'
raw_articles = load_articles_from_txt(articles_file_path)

# Extract data & list of dates from articles
articles, dates = extract_article_data(raw_articles)

# Load dictionaries & calculate sentiments
positive_dict_path = "Loughran-McDonald_Positive.csv"
negative_dict_path = "Loughran-McDonald_Negative.csv"
positive_dict = load_csv(positive_dict_path)
negative_dict = load_csv(negative_dict_path)
get_sentiment_scores(articles, positive_dict, negative_dict)

# Initialise dict to store daily sentiment
daily_senitment = {}
for article in articles:
    daily_senitment[article.date] = []

# Add sentiments for each day
for article in articles:
    daily_senitment[article.date].append(article.sentiment)
    
# Average sentiments for each day
for article in articles:
    daily_senitment[article.date] = np.mean(daily_senitment[article.date])

# Extract financial data from the time period
start_date = min(dates)
end_date = max(dates)
close_prices = ectract_close_prices("RYAAY.csv", start_date, end_date)
trading_days = get_trading_day_data(daily_senitment, close_prices)

# Save trading day data to csv
csv_file_path = 'trading_days_data.csv'
save_trading_days_to_csv(trading_days, csv_file_path)

# Variables for plot
dates = list(trading_days.keys())
sentiments = [trading_days[date].sentiment for date in trading_days]
closes = [trading_days[date].close for date in trading_days]
returns = [trading_days[date].returns for date in trading_days]

# Creating line plot
plt.plot(dates, returns, color='red', label='Returns')
plt.plot(dates, sentiments, label='Sentiment')
plt.title('Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# Some testing stats
if mode == "test":
    print("Positive word matches:")
    for word in positive_dict:
        if word in articles[0].body:
            print(word, sep=' ')
    print("\n")
            
    print("Negative word matches:")
    for word in negative_dict:
        if word in articles[0].body:
            print(word, sep=' ')
    print("\n")
    
    print(f"Headline: {articles[0].headline}\n")
    print(f"Date: {articles[0].date}\n")
    print(f"Body: {articles[0].body}\n")
    print(f"Sentiment: {articles[0].sentiment}\n")
    
    print(daily_senitment)
    print(trading_days)


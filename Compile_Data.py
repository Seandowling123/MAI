"""
Title: Compile Data
Author: Sean Dowling
Date: 15/04/2024

Description:
This script creates and aggregates the time series data required for the project. Its primary functions are to filter and process news
articles, calculate a time series of negative sentiment and calculate a time series of financial returns. These time series are aggregated
into a csv file and saved in the current directory.

Usage:
The script can be run without interaction, povided that the relevant inputs are available.

Inputs:
- News text data (stored as Raw_Articles/Articles_combined.txt)
- A list of valid sources (stored as Article_Data/News_Source_Names.csv) 
- Historical stock data (stored as Financial_Data/RYAAY.csv)
- Sentiment dictionary category and doamin-specific glossary (stored as Dictionaries_and_Glossaries/GI_Negative.csv 
    and Dictionaries_and_Glossaries/Combined_Glossary.csv)
  
  (Optional)
- Cached processed Article objects (stored as Article_Data/Articles_backup.pkl)
- Cached processed Article objects with a calculated sentiment proxy value (stored as Article_Data/Articles_backup_with_sentiment.pkl)

Outputs:
- Aggregated time series data (stored as Aggregated_Time_Series.csv)
- Extracted news article data (stored as Article_Data/Article_Data.csv) 

Dependencies:
- numpy (imported as np)
- pandas (imported as pd)
- math
- csv
- time
- re
- os
- statistics
- datetime and timedelta from the datetime module
- pickle
- word_tokenize from the nltk.tokenize module
- defaultdict from the collections module

"""


import numpy as np
import pandas as pd
import math
import csv
import time
import re
import os
import statistics
from datetime import datetime, timedelta
import pickle
from nltk.tokenize import word_tokenize
from collections import defaultdict

# Data to save for each trading day
class Trading_Day:
    def __init__(self, date, close, returns, neg_sentiment, 
                 article_count, monday, january, gfc, covid):
        self.date = date
        self.close = close
        self.returns = returns
        self.neg_sentiment = neg_sentiment
        self.article_count = article_count
        self.monday = monday
        self.january = january
        self.gfc = gfc
        self.covid = covid
    
    # Write daily data to csv
    def to_csv_line(self): 
        return (f"{str(self.date)},{str(self.close)},{str(10000*self.returns)},{str(abs(10000*self.returns))},"
        f"{str(self.neg_sentiment)},{str(self.article_count)},{str(self.monday)},{str(self.january)},{str(self.gfc)},"
        f"{str(self.covid)}")

# Class containing info about each article
class Article:
    def __init__(self, date, body, source, headline, neg_sentiment):
        self.date = date
        self.body = body
        self.source = source
        self.headline = headline
        self.neg_sentiment = neg_sentiment
        
# Class containing info about each news source
class Source:
    def __init__(self, names):
        self.name = names[0]
        if len(names) > 1:
            self.brands = names[1:len(names)]
        else: 
            print(f"Error loading source: {self.name}")
            self.brands = 0
        self.article_count = 0
        
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
    
# Load source names from csv
def load_source_names(file_path):
    try:
        sources = {}
        with open(file_path, 'r', newline='', encoding='utf-8-sig') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                sources[row[0]] = Source(row)
        return sources
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
def process_text(raw_article):
    try:
        # Extract article body & filter content
        body = (((raw_article.split("\nBody\n")[1]).split('Load-Date:')[0]).split("\nNotes\n")[0])
        body = body.replace('\n', ' ')
        body_filtered = re.sub(r'[^a-zA-Z ]', ' ', body)
        body_upper = body_filtered.upper()
        return body_upper
    except Exception as e:
        print("Error processing text")
        return 0, 0

# Find a string matching the date pattern
def get_date_match(article):
    date_pattern = re.compile(r'\n\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b \d{1,2}, (?:20|19)\d{2}')
    match = date_pattern.search(article.split("\nBody\n")[0])
    if match:
        date_string = match.group().replace('\n', '')
        date = convert_string_to_datetime(date_string)
        return date
    else: return 0

# Returns a dateTime object for news articles
def convert_string_to_datetime(date_string):
    try:
        datetime_object = datetime.strptime(date_string, '%B %d, %Y')
        return datetime_object
    except ValueError:
        print(f"Unable to parse the date string: {date_string}")
        return f"Unable to parse the date string: {date_string}"

# Find a string matching a source name
def get_source_match(article, sources):
    for source in sources.values():
        for source_name in source.brands:
            source_pattern = re.compile(r''+source_name+r'', re.IGNORECASE)
            match = source_pattern.search(article.split("\nBody\n")[0])
            # return recognised source and increase its count  
            if match:
                source_string = match.group().replace('\n', '')
                source.article_count = source.article_count+1
                return source_string
    return 0

# Returns a list of articles with the duplicates removed
def remove_duplicates(articles):
    return list(set(articles))

# Print a progress bar during calculations
def print_progress_bar(iteration, total, caption="Loading", bar_length=50):
    progress = iteration/total
    arrow = '-' * int(progress * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\r{caption}: [{arrow + spaces}] {progress:.2%}', end='', flush=True)

# Extracts date and body of each news article
def extract_article_data(raw_articles, sources, articles_backup_path):
    articles = []
    num_invalid_dates = 0
    num_out_of_date_range = 0
    num_invalid_sources = 0
    num_invalid_bodies = 0
    calculated = 0
    
    # Remove any duplicate articles
    len_orig = len(raw_articles)
    raw_articles = remove_duplicates(raw_articles).copy()
    num_duplicates = len_orig - len(raw_articles)
    
    # Extract data
    for i in range(len(raw_articles)):
        headline = raw_articles[i].split("\n")[1]
        
        # Check for valid date & source
        if "\nBody\n" in raw_articles[i]:
            
            # Find date pattern
            date = get_date_match(raw_articles[i])
            if isinstance(date, datetime):
                if date >= datetime(2004, 1, 1):
                    source = get_source_match(raw_articles[i], sources)
                    if source:
                        
                        # Add to Articles list. Initialise sentiment to 0
                        body = process_text(raw_articles[i])
                        if body != 0:
                            articles.append(Article(date, body, source, headline, 0))
                        else: num_invalid_bodies = num_invalid_bodies+1
                    else: num_invalid_sources = num_invalid_sources+1
                else: num_out_of_date_range = num_out_of_date_range+1
            else: num_invalid_dates = num_invalid_dates+1
        else: num_invalid_bodies = num_invalid_bodies+1
        
        # Progress tracker
        calculated = calculated + 1
        if (calculated % 10) == 0:
            print_progress_bar(calculated, len(raw_articles), caption="Processing Articles")
        
    # Print stats
    print(f"\nReceived {len_orig} articles.")
    print(f"Removed {num_duplicates} duplicate articles.")
    print(f"Removed {num_invalid_dates} articles with invalid dates.")
    print(f"Removed {num_out_of_date_range} articles out of date range.")
    print(f"Removed {num_invalid_sources} articles with invalid sources.")
    print(f"Removed {num_invalid_bodies} articles with invalid article bodies.\n")
    print(f"Loaded {len(articles)} articles.\n")
    articles_sum = 0
    print("Sources:")
    for source in sources: 
        print(f"{sources[source].name}: {sources[source].article_count}")
        articles_sum = articles_sum + sources[source].article_count
    print(f"TOTAL: {articles_sum}\n")

    # Save the article data to the backup file
    with open(articles_backup_path, 'wb') as file:
        pickle.dump(articles, file)
        
    return articles

# Count the number of base and domain dictionary words in an article
def get_word_count(text_body, dictionary_words, glossary):
    base_freq = 0
    domain_freq = 0
    article_words = word_tokenize(text_body)
    for word in list(set(dictionary_words + glossary)):
        if word in article_words:
            if word in dictionary_words and word in glossary:
                count = article_words.count(word)
                domain_freq += count
            elif word in dictionary_words and word not in glossary:
                count = article_words.count(word)
                base_freq += count
            elif word not in dictionary_words and word in glossary:
                count = article_words.count(word)
                domain_freq += count
    return base_freq, domain_freq

# Calculate sentiment score for text
def calculate_sentiment(text_body, negative_dict, glossary):
    # Get counts
    num_words = len(word_tokenize(text_body))
    neg_word_count, airline_term_count = get_word_count(text_body, negative_dict, glossary)
    
    # Calculate relative word frequencies
    neg_score = neg_word_count/num_words
    return neg_score

# Save sentiment score for an article
def save_sentiment_score(article, neg_sentiment):
    article.neg_sentiment = neg_sentiment
    
def get_sentiment_backup(seniment_backup_path):
    if os.path.exists(seniment_backup_path):
        with open(seniment_backup_path, 'rb') as file:
            loaded_object = pickle.load(file)
        return loaded_object
    else:
        print("No sentiment backup file found.")
        return None

# Calculate sentiment score
def get_sentiment_scores(articles, negative_dict, glossary, seniment_backup_path):
    calculated = 0
    num_articles = len(articles)
    
    # Check for valid sentiment backup file or calculate sentiment
    backedup_articles = get_sentiment_backup(seniment_backup_path)
    if backedup_articles != None and len(backedup_articles) == len(articles):
        print("Loading sentiments from backup file.")
        articles = backedup_articles.copy()
        print("Sentiments loaded from backup file.")
    else:
        # Iterate through each article
        total_time = 0
        for article in articles:
            try:
                # Get sentiment scores
                start_time = time.time()
                neg_sentiment = calculate_sentiment(article.body, negative_dict, glossary)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time = total_time+elapsed_time
                
                # Save score
                save_sentiment_score(article, neg_sentiment)
                    
                # Progress tracker
                calculated = calculated + 1
                if (calculated % 10) == 0:
                    print_progress_bar(calculated, num_articles, caption="Calculating Sentiment")
                
            except Exception as e:
                print(f" A sentiment calculation error occurred: {str(e)}\n")
        
        print("AVERAGE TIME:", total_time/len(articles))
        
        # Save the article data with sentiments to the backup file
        with open(seniment_backup_path, 'wb') as file:
            pickle.dump(articles, file)
            
    return articles

# Save the Article objects to csv
def save_article_data(articles, article_data_path):
    field_names = ["Date", "Source", "Headline", "Body", "Negative Sentiment"]
    
    # Write articles to CSV file
    with open(article_data_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(field_names)
        
        # Write data for each articles
        for article in articles:
            writer.writerow([
                article.date,
                article.source,
                article.headline,
                re.sub(r'\s{2,}', ' ', article.body),
                article.neg_sentiment
            ]) 

# Compute log of each value in a list   
def get_logs(input_list):
    logs = []
    for num in input_list:
        logs.append(math.log10(int(num)))
    return logs

# Convert the raw sentiments to Z-scores
def convert_to_zscore(daily_neg_sentiment):
    
    # Calculate the mean & standard deviation
    mean_neg = statistics.mean(list(daily_neg_sentiment.values()))
    std_dev_neg = statistics.stdev(list(daily_neg_sentiment.values()))
    
    # Convert sentiments to Z-scores
    for date in daily_neg_sentiment:
        daily_neg_sentiment[date] = (daily_neg_sentiment[date] - mean_neg) / std_dev_neg
    return daily_neg_sentiment

# Calculate a sentiment time series from the article sentiments
def get_daily_sentiments(articles):
    daily_neg_sentiment = defaultdict(list)
    daily_article_count = {}

    # Add sentiments for each day
    for article in articles:
        daily_neg_sentiment[article.date].append(article.neg_sentiment)
        
        # Update media volume for that day
        if article.date in daily_article_count:
            daily_article_count[article.date] = daily_article_count[article.date]+1
        else: daily_article_count[article.date] = 1
        
    # Average sentiments for each day
    for article in articles:
        daily_neg_sentiment[article.date] = np.mean(daily_neg_sentiment[article.date])
    
    # Convert time series to Z-score
    daily_neg_sentiment = convert_to_zscore(daily_neg_sentiment)
    return daily_neg_sentiment, daily_article_count

# Extract Ryanair financial data 
def get_RYAAY_data(file_path, start_date, end_date):
    close_price_dict = {}
    range_reached = 0
    index = 0
    try:
        with open(file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)
            
            # Calculate a closing price time series
            for row in reader:
                date_str = row['Date']
                date_object = datetime.strptime(date_str, '%Y-%m-%d')
                close_price = float(row['Adj Close'])
                if start_date <= date_object <= end_date:
                    
                    # Add the date before range for use in returns calculation
                    if range_reached == 0:
                        close_price_dict[prev_date] = prev_close
                        range_reached = 1
                    close_price_dict[date_object] = close_price
                    
                prev_date = date_object
                prev_close = close_price
                index = index+1
                
        print("RYAAY data compiled.\n")
        return close_price_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred. RYAAY data: {str(e)}")
        return None

# Get the date of the previous trading day
def get_previous_trading_day(day, trading_data):
    previous_day = day - timedelta(days=1)
    while previous_day not in trading_data:
        previous_day -= timedelta(days=1)
    return previous_day

# Get the sentiment z-score of days with no sentiment oriented words
def zero_match_sentiment(daily_neg_sentiment):
    mean = statistics.mean(list(daily_neg_sentiment.values()))
    std_dev = statistics.stdev(list(daily_neg_sentiment.values()))
    zero_match_z_score = (0 - mean) / std_dev
    return zero_match_z_score
    
# Check if a date is a Monday
def is_monday(date):
    if date.weekday() == 0:
        return 1
    else: return 0
    
# Check if a date is in January
def is_january(date):
    if date.month == 1:
        return 1
    else: return 0

# Check if a date occurs during global financial crisis
def is_gfc(date):
    # Set dates for global financial crisis
    gfc_start_date = datetime(2007, 12, 1)
    gfc_end_date = datetime(2009, 6, 30)
    
    # Check if date occurs during crash
    if gfc_start_date <= date <= gfc_end_date:
        return 1
    return 0

# Check if a date occurs during COVID-19 crash
def is_covid(date):
    # Set dates for COVID-19 crash
    covid_start_date = datetime(2020, 2, 1)
    covid_end_date = datetime(2020, 4, 30)
    
    # Check if date occurs during crash
    if covid_start_date <= date <= covid_end_date:
        return 1
    return 0
    
# Collect data for each trading day start_date, end_date
def aggregate_time_series(daily_neg_sentiment, daily_article_count, close_prices):
    daily_data = {}
    returns_list = []
    days_parsed = 0
    
    # Calculate sentiment for missing data days
    #base_sentiment = zero_match_sentiment(daily_neg_sentiment)
    base_sentiment = min(daily_neg_sentiment.values())
    
    # Iterate through dates and compile the data
    for date in close_prices:
        close = 0
        returns = 0
        neg_sentiment = base_sentiment
        article_count = 0
        monday = 0
        january = 0
        gfc = 0
        covid = 0
        
        # Skip the first trading day date
        if days_parsed == 0:
            days_parsed = 1
        else:
            # Collect financial data
            close = close_prices[date]
            returns = math.log(close_prices[date]/close_prices[get_previous_trading_day(date, close_prices)])
            returns_list.append(returns)
            
            # Collect calendar data
            monday = is_monday(date)
            january = is_january(date)
            gfc = is_gfc(date)
            covid = is_covid(date)
            
            # Collect sentiment data
            if date in daily_neg_sentiment:
                neg_sentiment = daily_neg_sentiment[date]
                article_count = daily_article_count[date]
                
            # Save all data 
            daily_data[date] = Trading_Day(date, close, returns, neg_sentiment,
                                           article_count, monday, january, gfc, covid)
        
    print("Financial time series compiled.\n")
    return daily_data

# Save collected data to csv
def save_time_series_to_csv(daily_data, csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Header
            writer.writerow(["Date","Close","Returns","Absolute_Returns","Negative_Sentiment",
                             "Article_Count","Monday","January","GFC","COVID"])
            # Save data
            for date, trading_day in daily_data.items():
                writer.writerow(trading_day.to_csv_line().split(','))

        print(f"Aggregated time series saved to {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Article file paths
article_data_path = 'Raw_Articles/Articles_combined.txt'
articles_backup_path = 'Article_Data/Articles_backup.pkl'
seniment_backup_path = "Article_Data/Articles_backup_with_sentiment.pkl"
article_data_backup_path = "Article_Data/Article_Data.csv"
sources_data_path = "Article_Data/News_Source_Names.csv"

# Financial data file paths
RYAAY_data_path = "Financial_Data/RYAAY.csv"

# Dictionaries file paths
negative_dict_path = "Dictionaries_and_Glossaries/GI_Negative.csv"
glossary_path = "Dictionaries_and_Glossaries/Combined_Glossary.csv"

# Output time series file path
output_series_file_path = "Aggregated_Time_Series.csv"

# Check for backup and load files
if os.path.exists(articles_backup_path):
    print("Loading articles from backup file.")
    with open(articles_backup_path, 'rb') as file:
        articles = pickle.load(file)
        print(f"Loaded {len(articles)} articles from backup file.")
else:  
    # Extract data & list of dates from the articles
    sources = load_source_names(sources_data_path)
    raw_articles = load_articles_from_txt(article_data_path)
    articles = extract_article_data(raw_articles, sources, articles_backup_path)

# Load dictionaries & calculate sentiments
negative_dict = load_csv(negative_dict_path)
glossary = load_csv(glossary_path)
articles = get_sentiment_scores(articles, negative_dict, glossary, seniment_backup_path)

# Save the article data to csv
save_article_data(articles, article_data_backup_path)

# Get sentiment time series    
daily_neg_sentiment, daily_article_count = get_daily_sentiments(articles)

# Get the time period
dates = [article.date for article in articles]
start_date = min(dates)
end_date = max(dates)
print(f"\nTime series start date: {start_date}, end date: {end_date}\n")

# Extract financial data from the time period
close_prices= get_RYAAY_data(RYAAY_data_path, start_date, end_date)
daily_data = aggregate_time_series(daily_neg_sentiment, daily_article_count, close_prices)

# Save data to csv
save_time_series_to_csv(daily_data, output_series_file_path)
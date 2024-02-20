import numpy as np
import pandas as pd
import math
import csv
import time
import re
import os
import statistics
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from collections import defaultdict
#nltk.download('punkt')

# Data to save for each trading day
class Trading_Day:
    def __init__(self, date, close, returns, absolute_returns, volume, vix_returns, vix_close, monday, january, sentiment, stemmed_sentiment, media_volume):
        self.date = date
        self.close = close
        self.returns = returns
        self.absolute_returns = absolute_returns
        self.volume = volume
        self.vix_returns = vix_returns
        self.vix_close = vix_close
        self.sentiment = sentiment
        self.stemmed_sentiment = stemmed_sentiment
        self.media_volume = media_volume
        self.monday = monday
        self.january = january
    
    def to_csv_line(self):
        return f"{str(self.date)},{str(self.close)},{str(self.returns)},{str(self.absolute_returns)},{str(self.volume)},{str(self.vix_close)},{str(self.vix_returns)},{str(self.sentiment)},{str(self.stemmed_sentiment)},{str(self.media_volume)},{str(self.monday)},{str(self.january)}"

# Class containing info about each article
class Article:
    def __init__(self, date, body, stemmed_body, source, headline, sentiment, stemmed_sentiment):
        self.date = date
        self.body = body
        self.stemmed_body = stemmed_body
        self.source = source
        self.headline = headline
        self.sentiment = sentiment
        self.stemmed_sentiment = stemmed_sentiment
        
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
        with open(file_path, 'r', newline='') as csv_file:
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

# Converts words in an article body to their stems
def stem_text(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_text = ""
    
    # Stem each word
    for word in words:
        stemmed_word = stemmer.stem(word)
        stemmed_text = stemmed_text + " " + stemmed_word
    #print(text, stemmed_text)
    return stemmed_text.upper()

# Extracts date and body of each news article
def extract_article_data(raw_articles, sources, articles_backup_path):
    articles = []
    dates = []
    num_invalid_dates = 0
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
        
        # Find date pattern
        if "\nBody\n" in raw_articles[i]:
            date = get_date_match(raw_articles[i])
            
            # Check for valid date & source
            if isinstance(date, datetime):
                source = get_source_match(raw_articles[i], sources)
                if source:
                    
                    # Add to Articles list. Initialise senitment to 0
                    dates.append(date)
                    body = process_text(raw_articles[i])
                    stemmed_body = stem_text(body)
                    if body != 0:
                        articles.append(Article(date, body, stemmed_body, source, headline, 0, 0))
                    else: num_invalid_bodies = num_invalid_bodies+1
                else: num_invalid_sources = num_invalid_sources+1
            else: num_invalid_dates = num_invalid_dates+1 
        else: num_invalid_bodies = num_invalid_bodies+1
        
        # Progress tracker
        calculated = calculated + 1
        progress = "{:.2f}".format((calculated/len(raw_articles))*100)
        if (calculated % 10) == 0:
            print(f"Loading articles: {progress}%\r", end='', flush=True)
        
    # Print stats
    print(f"Received {len(raw_articles)} articles.")
    print(f"Removed {num_duplicates} duplicate articles.")
    print(f"Removed {num_invalid_dates} articles with invalid dates.")
    print(f"Removed {num_invalid_sources} articles with invalid sources.")
    print(f"Removed {num_invalid_bodies} articles with invalid article bodies.\n")
    print(f"Loaded {len(articles)} articles.\n")
    articles_sum = 0
    for source in sources: 
        print(f"{sources[source].name}: {sources[source].article_count}")
        articles_sum = articles_sum + sources[source].article_count
    print(f"TOTAL: {articles_sum}\n")

    # Save the article data to the backup file
    with open(articles_backup_path, 'wb') as file:
        pickle.dump((articles, dates), file)
        
    return articles, dates

# Load article sentiments from backup file
def Load_senitments_from_backup(articles, seniment_backup_path):
    if os.path.exists(seniment_backup_path):
        sentiments = []
        stemmed_sentiments = []
        print(f"Loading sentiments from backup file: {seniment_backup_path}.")
        
        # Open the csv file and read its contents
        with open(seniment_backup_path, 'r', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                sentiments.append(row[0])
                stemmed_sentiments.append(row[1])
        
        for i in range(len(sentiments)):
            articles[i].sentiment = float(sentiments[i])
            articles[i].stemmed_sentiment = float(stemmed_sentiments[i])
        print(f"Loaded {len(sentiments)} sentiments from backup.\n")

# Convert the raw sentiments to Z-scores
def convert_to_zscore(articles):
    
    # Extract senitments
    sentiments = [article.sentiment for article in articles]
    stemmed_sentiments = [article.stemmed_sentiment for article in articles]

    # Calculate the mean & standard deviation
    mean = statistics.mean(sentiments)
    std_dev = statistics.stdev(sentiments)
    mean_stemmed = statistics.mean(stemmed_sentiments)
    std_dev_stemmed = statistics.stdev(stemmed_sentiments)
    
    # Convert senitments to Z-scores
    for article in articles:
        article.sentiment = (article.sentiment - mean) / std_dev
        article.stemmed_sentiment = (article.stemmed_sentiment - mean_stemmed) / std_dev_stemmed

# Count the number of dictionary words in an article
def get_word_count(article, word_list, glossary):
    word_counts = 0
    article_words = article.split()
    for word in word_list:
        # Check if the word appears in the glossary
        if word not in glossary:
            count = article_words.count(word)
            word_counts += count
    return word_counts

def calculate_sentiment(text_body, positive_dict, negative_dict, glossary):
    # Get counts
    num_words = len(word_tokenize(text_body))
    pos_word_count = get_word_count(text_body, positive_dict, glossary)
    neg_word_count = get_word_count(text_body, negative_dict, glossary)
    
    # Calculate relative word frequencies
    pos_score = pos_word_count
    neg_score = neg_word_count
    total_score = (pos_score - neg_score)/num_words
    
    return total_score

# Calculate sentiment score
def get_sentiment_scores(articles, positive_dict, negative_dict, glossary, seniment_backup_path):
    calculated = 0
    num_articles = len(articles)
    
    # Open first csv file
    with open(seniment_backup_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        
        # Iterate through each article
        for article in articles:
            try:
                # If the article has no senitment, calculate it
                if article.sentiment == 0 and article.stemmed_sentiment == 0:
                    sentiment = calculate_sentiment(article.body, positive_dict, negative_dict, glossary)
                    stemmed_sentiment  = calculate_sentiment(article.stemmed_body, positive_dict, negative_dict, glossary)
                
                    # Save score
                    article.sentiment = sentiment
                    article.stemmed_sentiment = stemmed_sentiment
                writer.writerow([article.sentiment, article.stemmed_sentiment])
                
                # Progress tracker
                calculated = calculated + 1
                progress = "{:.2f}".format((calculated/num_articles)*100)
                if (calculated % 10) == 0:
                    print(f"Calculating Sentiment: {progress}%\r", end='', flush=True)
                
            except Exception as e:
                print(f"An sentiment calculation error occurred: {str(e)}\n")
    
        # Convert the sentiments to Z-scores
        convert_to_zscore(articles)
            
# Compute log of each value in a list   
def get_logs(input_list):
    logs = []
    for num in input_list:
        logs.append(math.log10(int(num)))
    return logs

# Calculate detrended daily trading volume
def get_detrended_volume(volume, index):
    log_vol = get_logs(volume)
    mean_vol = np.mean(log_vol[index-60:index])
    detrended_vol = log_vol[index] - mean_vol
    return detrended_vol

# Calculate a sentiment time series from the article sentiments
def get_daily_sentiments(articles):
    daily_sentiment = defaultdict(list)
    daily_stemmed_sentiment = defaultdict(list)
    daily_media_volume = {}

    # Add sentiments for each day
    for article in articles:
        daily_sentiment[article.date].append(article.sentiment)
        daily_stemmed_sentiment[article.date].append(article.stemmed_sentiment)
        
        # Update media volume for that day
        if article.date in daily_media_volume:
            daily_media_volume[article.date] = daily_media_volume[article.date]+1
        else: daily_media_volume[article.date] = 1
        
    # Average sentiments for each day
    for article in articles:
        daily_sentiment[article.date] = np.mean(daily_sentiment[article.date])
        daily_stemmed_sentiment[article.date] = np.mean(daily_stemmed_sentiment[article.date])
        
    return daily_sentiment, daily_stemmed_sentiment, daily_media_volume

# Extract Ryanair financial data 
def get_RYAAY_data(file_path, start_date, end_date):
    close_price_dict = {}
    trading_vol_dict = {}
    volume = []
    range_reached = 0
    index = 0
    try:
        with open(file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)
            
            for row in reader:
                date_str = row['Date']
                date_object = datetime.strptime(date_str, '%Y-%m-%d')
                close_price = float(row['Adj Close'])
                volume.append(row['Volume'])

                if start_date <= date_object <= end_date:
                    # Add the date before range for use in returns calculation
                    if range_reached == 0:
                        close_price_dict[prev_date] = prev_close
                        range_reached = 1
                    close_price_dict[date_object] = close_price
                    
                    # Get detrended trading volume
                    detrended_vol = get_detrended_volume(volume, index)
                    trading_vol_dict[date_object] = detrended_vol
                    
                prev_date = date_object
                prev_close = close_price
                index = index+1
                
        print("RYAAY data compiled.\n")
        return close_price_dict, trading_vol_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred. RYAAY data: {str(e)}")
        return None
    
def get_VIX_data(file_path, start_date, end_date):
    close_price_dict = {}
    prev_date = 0
    range_reached = 0
    try:
        with open(file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)
            
            for row in reader:
                date_str = row['Date']
                date_object = datetime.strptime(date_str, '%Y-%m-%d')
                
                # If there is data for this day, then add it to the dict
                if row['Adj Close'] != "null":
                    close_price = float(row['Adj Close'])
                    if start_date <= date_object <= end_date:
                        # Add the date before range for use in returns calculation
                        if range_reached == 0:
                            close_price_dict[prev_date] = prev_close
                            range_reached = 1
                        close_price_dict[date_object] = close_price
                        
                    prev_date = date_object
                    prev_close = close_price
                else: close_price_dict[date_object] = prev_close
                
        print("VIX data compiled.\n")
        return close_price_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred. VIX data: {str(e)}")
        return None
    
# Get the date of the previous trading day
def get_previous_trading_day(day, trading_data):
    previous_day = day - timedelta(days=1)
    while previous_day not in trading_data:
        previous_day -= timedelta(days=1)
    return previous_day
    
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
    
# Collect data for each trading day start_date, end_date
def get_trading_day_data(daily_sentiment, daily_stemmed_sentiment, daily_media_volume, close_prices, trading_volume, VIX_prices):
    daily_data = {}
    first_date = 0
    
    # Iterate through dates and compile the data
    for date in close_prices:
        close = 0
        returns = 0
        volume = 0
        vix = 0
        monday = 0
        january = 0
        senitment = 0
        stemmed_sentiment = 0
        media_volume = 0
        
        # Skip the first trading day date
        if first_date == 0:
            first_date = 1
        else:
            # Collect data and store it in trading days dict
            close = close_prices[date]
            returns = math.log(close_prices[date]/close_prices[get_previous_trading_day(date, close_prices)])
            volume = trading_volume[date]
            vix_close = VIX_prices[date]
            vix = math.log(VIX_prices[date]/VIX_prices[get_previous_trading_day(date, VIX_prices)])
            monday = is_monday(date)
            january = is_january(date)
            if date in daily_sentiment:
                senitment = daily_sentiment[date]
                stemmed_sentiment = daily_stemmed_sentiment[date]
                media_volume = daily_media_volume[date]
            daily_data[date] = Trading_Day(date, close, returns, abs(returns), volume, vix, vix_close, monday, january, senitment, stemmed_sentiment, media_volume)
        
    print("Trading data compiled.\n")
    return daily_data

# Save collected data to csv
def save_daily_data_to_csv(daily_data, csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Header
            writer.writerow(["Date", "Close", "Returns", "Absolute_Returns", "Detrended_Volume", "VIX_Close", "VIX_Returns", "Sentiment","Stemmed_Sentiment", "Media_Volume", "Monday", "January",])
            # Save data
            for date, trading_day in daily_data.items():
                writer.writerow(trading_day.to_csv_line().split(','))

        print(f"Trading days data saved to {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Select mode
mode  = "tes"

#articles_file_path = 'Articles_txt/Financial(1001-1500).txt'
articles_file_path = 'Articles_txt_combined/Articles_combined.txt'
articles_backup_path = 'Articles_backup.pkl'
sources_file_path = 'News_Source_Names.csv'
seniment_backup_path = "sentiments_backup.csv"

# Check for backup and load files
if os.path.exists(articles_backup_path):
    with open(articles_backup_path, 'rb') as file:
        articles, dates = pickle.load(file)
        print(f"Loaded {len(articles)} articles from backup file.")
else:  
    # Extract data & list of dates from the articles
    sources = load_source_names(sources_file_path)
    raw_articles = load_articles_from_txt(articles_file_path)
    articles, dates = extract_article_data(raw_articles, sources, articles_backup_path)

# Load dictionaries & calculate sentiments
positive_dict_path = "Dictionaries_Glossaries/GI_Positive.csv"
negative_dict_path = "Dictionaries_Glossaries/GI_Negative.csv"
glossary_path = "Dictionaries_Glossaries/Combined_Glossary.csv"
positive_dict = load_csv(positive_dict_path)
negative_dict = load_csv(negative_dict_path)
glossary = load_csv(glossary_path)
print(glossary)
Load_senitments_from_backup(articles, seniment_backup_path)
get_sentiment_scores(articles, positive_dict, negative_dict, glossary, seniment_backup_path)

# Get sentiment time series    
daily_sentiment, daily_stemmed_sentiment, daily_media_volume = get_daily_sentiments(articles)

# Get the time period
start_date = min(dates)
end_date = max(dates)
print(f"Start date: {start_date} | End date: {end_date}\n")

# Extract financial data from the time period
close_prices, trading_volume = get_RYAAY_data("RYAAY.csv", start_date, end_date)
VIX_prices = get_VIX_data("VIX.csv", start_date, end_date)
daily_data = get_trading_day_data(daily_sentiment, daily_stemmed_sentiment, daily_media_volume, close_prices, trading_volume, VIX_prices)

# Save data to csv
daily_csv_file_path = 'XX_daily_data.csv'
save_daily_data_to_csv(daily_data, daily_csv_file_path)

# To plot daily or weekly data
plotting_variable = daily_data

# Variables for plot
dates = list(plotting_variable.keys())
sentiments = [plotting_variable[date].sentiment for date in plotting_variable]
returns = [plotting_variable[date].returns for date in plotting_variable]
volume = [plotting_variable[date].volume for date in plotting_variable]
vix = [plotting_variable[date].vix_close for date in plotting_variable]

# Creating line plot
plt.plot(dates, returns, color='red', label='Returns')
plt.plot(dates, sentiments, label='Sentiment')
#plt.plot(dates, vix, label='VIX')
plt.title('Values Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()


# Some testing stats
if mode == "test":
    for i in range(0, len(articles), 1000):
        print("Positive word matches:")
        for word in positive_dict:
            if word in articles[0].body:
                print(word, sep=' ')
        print("\n")
                
        print("Negative word matches:")
        for word in negative_dict:
            if word in articles[0].body:
                print(word, sep=' ')
        print("-------------------------------------------------\n")
    
    print(f"Headline: {articles[0].headline}\n")
    print(f"Date: {articles[0].date}\n")
    print(f"Body: {articles[0].body}\n")
    print(f"Sentiment: {articles[0].sentiment}\n")


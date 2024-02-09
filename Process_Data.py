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
from nltk.corpus import sentiwordnet as swn
nltk.download('punkt')

# Data to save for each trading day
class Trading_Day:
    def __init__(self, date, close, returns, absolute_returns, volume, vix, monday, january, sentiment):
        self.date = date
        self.close = close
        self.returns = returns
        self.absolute_returns = absolute_returns
        self.volume = volume
        self.vix = vix
        self.monday = monday
        self.january = january
        self.sentiment = sentiment
    
    def to_csv_line(self):
        return f"{str(self.date)},{str(self.close)},{str(self.returns)},{str(self.absolute_returns)},{str(self.volume)},{str(self.vix)},{str(self.monday)},{str(self.january)},{str(self.sentiment)}"

# Class containing info about each article
class Article:
    def __init__(self, date, body, source, headline, sentiment):
        self.date = date
        self.body = body
        self.source = source
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
        body_split = (((body.split("\nBody\n")[1]).split('Load-Date:')[0]).split("\nNotes\n")[0])
        body_split = (body_split.replace('\n', '')).replace('  ', '')
        body_filtered = re.sub(r'[^a-zA-Z ]', '', body_split)
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

# Get the list of all article sources 
def get_sources_list():
    return ["Wall Street Journal Abstracts","Financial Times Online","Associated Press Financial Wire","Airguide Business & AirguideBusiness\.com","Proactive Investors \(UK\)","BNS News Service in English by Baltic News Service \(BNS\) English","Newstex Blogs","Live Briefs PRO Global Markets","MT Newswires Live Briefs","Business World \(Digest\)","MarketLine NewsWire","London Stock Exchange Regulatory News Service","Sunday Business Post","International Business Times News","The Investors Chronicle","Financial Times \(London, England\)","AirFinance Journal","Flight International","dpa-AFX International ProFeed","dpa international \(Englischer Dienst\)","RTT News \(United States\)","Citywire","City A\.M\.","ANSA English Corporate Service","American Banking and Market News","Transcript Daily","Watchlist News","DailyPolitical","Alliance News UK","Thomson Financial News Super Focus"]

# Find a string matching a source name
def get_source_match(article):
    sources = get_sources_list()
    for source in sources:
        source_pattern = re.compile(r''+source+r'')
        match = source_pattern.search(article.split("\nBody\n")[0])
        if match:
            source_string = match.group().replace('\n', '')
            return source_string 
    return 0

# Extracts date and body of each news article
def extract_article_data(raw_articles):
    articles = []
    dates = []
    num_invalid_dates = 0
    num_invalid_sources = 0
    num_invalid_bodies = 0
    
    # Extract data
    for i in range(len(raw_articles)):
        headline = raw_articles[i].split("\n")[1]
        
        # Find date pattern
        if "\nBody\n" in raw_articles[i]:
            date = get_date_match(raw_articles[i])
            source = get_source_match(raw_articles[i])
            
            if not source:
                print(raw_articles[i].split("\nBody\n")[0])
            
            # Check for valid date & source
            if date and source:
                
                # Add to Articles list. Initialise senitment to 0
                if isinstance(date, datetime):
                    dates.append(date)
                    body = process_text(raw_articles[i])
                    if body != 0:
                        articles.append(Article(date, body, source, headline, 0))
                    else: num_invalid_bodies = num_invalid_bodies+1
                else: num_invalid_dates = num_invalid_dates+1
            else: num_invalid_dates = num_invalid_dates+1
        else: num_invalid_bodies = num_invalid_bodies+1
        
    
    print(f"Received {len(raw_articles)} articles.")
    print(f"Removed {num_invalid_dates} articles with invalid dates.")
    print(f"Removed {num_invalid_bodies} articles with invalid article bodies.")
    print(f"Loaded {len(articles)} articles.\n")
    return articles, dates

# Load article sentiments from backup file
def Load_senitments_from_backup(articles, seniment_backup_path):
    if os.path.exists(seniment_backup_path):
        print(f"Loading sentiments from backup file: {seniment_backup_path}.")
        sentiments_from_backup = load_csv(seniment_backup_path)
        for i in range(len(sentiments_from_backup)):
            articles[i].sentiment = float(sentiments_from_backup[i])
        print(f"Loaded {len(sentiments_from_backup)} sentiments from backup.\n")
        
# Count the number of dictionary words in an article
def get_word_count(article, word_list):
    word_counts = 0
    for word in tuple(word_list):
        count = article.count(word)
        word_counts = word_counts + count
    return word_counts

# Calculate sentiment score
def get_sentiment_scores(articles, positive_dict, negative_dict, seniment_backup_path):
    calculated = 0
    num_articles = len(articles)
    with open(seniment_backup_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for article in articles:
            try:
                if article.sentiment == 0:
                    # Get counts
                    num_words = len(word_tokenize(article.body))
                    pos_word_count = get_word_count(article.body, positive_dict)
                    neg_word_count = get_word_count(article.body, negative_dict)
                    
                    # Calculate relative word frequencies
                    pos_score = pos_word_count
                    neg_score = neg_word_count
                    total_score = (pos_score - neg_score)/num_words
                
                    # Save score
                    article.sentiment = total_score
                writer.writerow([article.sentiment])
                
                # Progress tracker
                calculated = calculated + 1
                progress = "{:.2f}".format((calculated/num_articles)*100)
                if (calculated % 10) == 0:
                    print(f"Calculating Sentiment: {progress}%\r", end='', flush=True)
                
            except Exception as e:
                print(f"An sentiment calculation error occurred: {str(e)}\n")
            
# Compute log of each value in a list   
def get_logs(input_list):
    logs = []
    for num in input_list:
        logs.append(math.log10(int(num)))
    return logs

# Extract financial data 
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
                    # Add the date before range for returns calculation
                    if range_reached == 0:
                        close_price_dict[prev_date] = prev_close
                        range_reached = 1
                    close_price_dict[date_object] = close_price
                    
                    # Get detrended trading volume
                    detrended_vol = np.mean(get_logs(volume[index-60:index]))
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
    try:
        with open(file_path, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)
            
            for row in reader:
                date_str = row['Date']
                if row['Adj Close'] != "null":
                    date_object = datetime.strptime(date_str, '%Y-%m-%d')
                    close_price = float(row['Adj Close'])

                    if start_date <= date_object <= end_date:
                        close_price_dict[date_object] = close_price
                else: close_price_dict[date_object] = 0
                
        print("VIX data compiled.\n")
        return close_price_dict
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred. VIX data: {str(e)}")
        return None
    
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

# Collect data for each trading day
def get_trading_day_data(daily_senitment, close_prices, trading_volume, VIX_prices):
    trading_days = {}
    prev_date = 0
    for date in close_prices:
        if prev_date != 0:
            # Compile data
            close = close_prices[date]
            returns = math.log(close_prices[date]/close_prices[prev_date])
            monday = is_monday(date)
            january = is_january(date)
            volume = trading_volume[date]
            vix = VIX_prices[date]
            if date in daily_senitment:
                senitment = daily_senitment[date]
            else: senitment = 0
            # Store in trading days dict
            trading_days[date] = Trading_Day(date, close, returns, abs(returns), volume, vix, monday, january, senitment)
        prev_date = date
    print("Trading Days data compiled.\n")
    return trading_days

def save_trading_days_to_csv(trading_days, csv_file_path):
    try:
        with open(csv_file_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            # Header
            writer.writerow(["Date", "Close", "Returns", "Absolute_Returns", "Detrended_Volume", "VIX", "Monday", "January", "Sentiment"])
            # Save data
            for date, trading_day in trading_days.items():
                writer.writerow(trading_day.to_csv_line().split(','))

        print(f"Trading days data saved to {csv_file_path}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Select mode
mode  = "tes"

articles_file_path = 'Articles_txt/Financial(1001-1500).txt'
#articles_file_path = 'Articles_txt_combined/Articles_combined.txt'
seniment_backup_path = "sentiments_backup.csv"

# Load files
#articles_file_path = 'Sample_article.txt'
raw_articles = load_articles_from_txt(articles_file_path)

# Extract data & list of dates from articles
articles, dates = extract_article_data(raw_articles)

# Load dictionaries & calculate sentiments
#positive_dict_path = "Loughran-McDonald_Positive.csv"
#negative_dict_path = "Loughran-McDonald_Negative.csv"
positive_dict_path = "GI_Positive.csv"
negative_dict_path = "GI_Negative.csv"
positive_dict = load_csv(positive_dict_path)
negative_dict = load_csv(negative_dict_path)
Load_senitments_from_backup(articles, seniment_backup_path)
get_sentiment_scores(articles, positive_dict, negative_dict, seniment_backup_path)

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
print(f"Start date: {start_date} End date: {end_date}\n")
close_prices, trading_volume = get_RYAAY_data("RYAAY.csv", start_date, end_date)
VIX_prices = get_VIX_data("VIX.csv", start_date, end_date)
trading_days = get_trading_day_data(daily_senitment, close_prices, trading_volume, VIX_prices)

# Save trading day data to csv
csv_file_path = 'trading_days_data.csv'
save_trading_days_to_csv(trading_days, csv_file_path)

# Variables for plot
dates = list(trading_days.keys())
sentiments = [trading_days[date].sentiment for date in trading_days]
closes = [trading_days[date].close for date in trading_days]
returns = [trading_days[date].returns for date in trading_days]
volume = [trading_days[date].volume for date in trading_days]
vix = [trading_days[date].vix for date in trading_days]

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


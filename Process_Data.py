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

# Class containing info about each article
class Article:
    def __init__(self, date, body, sentiment):
        self.date = date
        self.body = body
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
        print(body_filtered, "\n")
        return body_filtered
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
def extract_article_data(text_data):
    articles = []
    dates = []
    # Extract data
    for i in range(len(text_data)):
        # Extract date
        date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b \d{1,2}, \d{4}')
        match = date_pattern.search(text_data[i])
        # Check for valid date
        if match:
            date_string = match.group()
            date = convert_string_to_datetime(date_string)
            if isinstance(date, datetime):
                dates.append(date)
                # Process text body
                body = process_text(text_data[i])
                if body != 0:
                    articles.append(Article(date, body, 0))
                else: print("Removed article. Incorrect syntax")
            else: print("Removed article. Date loaded incorrectly")
        else: print("Removed article. Date not found")
    print("Loaded", len(articles), "articles")
    return articles, dates

def get_sentiment_scores(articles, positive_dict, negative_dict):
    

# Load files
articles_file_path = 'Articles_txt_combined/Articles_combined.txt'
raw_articles = load_articles_from_txt(articles_file_path)

# Extract data & list of dates from articles
articles, dates = extract_article_data(raw_articles)

positive_dict_path = "Loughran-McDonald_Positive.csv"
negative_dict_path = "Loughran-McDonald_Negative.csv"
# Load dictionary from csv
positive_dict = load_csv(positive_dict_path)
negative_dict = load_csv(negative_dict_path)

get_sentiment_scores()


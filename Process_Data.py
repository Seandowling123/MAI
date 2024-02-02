from transformers import BertTokenizer, BertForSequenceClassification
import torch
import csv
from datetime import datetime
import re
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import time
import os

def load_text_file(file_path):
    try:
        with open(file_path, 'r', encoding="latin-1") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
def process_text(body):
    try:
        body_split = ((((body.split("\nBody\n")[1]).split('Load-Date:')[0]).split("\nNotes\n")[0]).replace('\n', '')).replace('  ', '')
        body_filtered = re.sub(r'[^a-zA-Z ]', '', body_split)
        return body_filtered
    except Exception as e:
        print("Error processing text")
        return 0, 0
    
def convert_string_to_datetime(date_string):
    try:
        datetime_object = datetime.strptime(date_string, '%B %d, %Y')
        return datetime_object
    except ValueError:
        return f"Unable to parse the date string: {date_string}"
    
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



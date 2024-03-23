import re
from nltk.tokenize import word_tokenize
import csv

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

# Count the number of dictionary words in an article
def get_words(text_body, dictionary_words, glossary):
    words = []
    word_counts = 0
    article_words = word_tokenize(text_body)
    for word in dictionary_words: 
        # Check if the word appears in the glossary
        if word not in glossary:
            count = article_words.count(word)
            word_counts += count
            if article_words.count(word) > 0:
                words.append(word)
    return word_counts/len(article_words), len(article_words), words


article_data_path = 'Raw_Articles/Articles_combined.txt'

# Dictionaries file paths
positive_dict_path = "Dictionaries_and_Glossaries/GI_Positive.csv"
negative_dict_path = "Dictionaries_and_Glossaries/GI_Negative.csv"
glossary_path = "Dictionaries_and_Glossaries/Combined_Glossary.csv"

positive_dict = load_csv(positive_dict_path)
negative_dict = load_csv(negative_dict_path)
glossary = load_csv(glossary_path)

raw_articles = load_articles_from_txt(article_data_path)

for article in raw_articles:
    positive = get_words(process_text(article), positive_dict, glossary)
    negative = get_words(process_text(article), negative_dict, glossary)
    if len(positive[2]) > 15:
        print(article.split("\n")[1])


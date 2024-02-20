import csv
import re

def get_words(file_path):
    words = []
    with open(file_path, 'r', encoding="latin-1") as file:
        for line_num, line in enumerate(file, start=1):
            words.append(line.split('\n')[0])
            print(line.split('\n')[0])
    return words

def save_words_to_csv(words, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for word in words:
            word = word.encode('utf-8', 'ignore').decode('utf-8')  # Encode to UTF-8 and ignore non-UTF-8 characters
            word_filtered = (word.split(' (')[0])
            word_filtered = re.sub(r'[^a-zA-Z0-9 ]', '', word_filtered)
            word_upper = word_filtered.upper()
            writer.writerow([word_upper])

words = get_words("britannica.txt")
save_words_to_csv(words, 'Britannica_Glossary.csv')



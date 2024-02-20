import csv
import re

def get_words(file_path):
    words = []
    with open(file_path, 'r', encoding="latin-1") as file:
        for line_num, line in enumerate(file, start=1):
            if line_num % 3 == 1:  # Check if it's every third line
                words.append(line.split('\n')[0])
                print(line.split('\n')[0])
    return words

def save_words_to_csv(words, file_path):
    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        for word in words:
            word = word.encode('utf-8', 'ignore').decode('utf-8')  # Encode to UTF-8 and ignore non-UTF-8 characters
            word_filtered = word.split('(')[0].replace((' ', ''))
            word_filtered = re.sub(r'[^a-zA-Z ]', ' ', word_filtered)
            word_upper = word_filtered.upper()
            writer.writerow([word_upper])

words = get_words("A4A_webpage.txt")
save_words_to_csv(words, 'A4A_Glossary.csv')



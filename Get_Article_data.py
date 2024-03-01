import re
import datetime
import csv

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

# Extracts date and body of each news article
def extract_article_data(raw_articles, sources, articles_backup_path):
    articles = []
    dates = []
    calculated = 0
    
    # Remove any duplicate articles
    len_orig = len(raw_articles)
    raw_articles = remove_duplicates(raw_articles).copy()
    num_duplicates = len_orig - len(raw_articles)
    
    # Extract data
    for i in range(len(raw_articles)):
        source = get_source_match(raw_articles[i], sources)
        
        # Progress tracker
        calculated = calculated + 1
        progress = "{:.2f}".format((calculated/len(raw_articles))*100)
        if (calculated % 10) == 0:
            print(f"Loading articles: {progress}%\r", end='', flush=True)
        
    # Print stats
    print(f"Received {len(raw_articles)} articles.")
    print(f"Removed {num_duplicates} duplicate articles.")
    print(f"Loaded {len(articles)} articles.\n")
    articles_sum = 0
    for source in sources: 
        print(f"{sources[source].name}: {sources[source].article_count}")
        articles_sum = articles_sum + sources[source].article_count
    print(f"TOTAL: {articles_sum}\n")
    
articles_file_path = 'Articles_txt_combined/Articles_combined.txt'
articles_backup_path = 'Articles_backup.pkl'
sources_file_path = 'News_Source_Names.csv'
seniment_backup_path = "sentiments_backup.csv"

# Extract data & list of dates from the articles
sources = load_source_names(sources_file_path)
raw_articles = load_articles_from_txt(articles_file_path)
articles, dates = extract_article_data(raw_articles, sources, articles_backup_path)
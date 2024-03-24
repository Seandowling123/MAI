import re
import csv
from datetime import datetime

# Load dictionary words from csv
def load_csv(file_path):
    try:
        with open(file_path, 'r', newline='') as csv_file:
            reader = csv.reader(csv_file)
            entries = [row for row in reader]
        return entries
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
def format_date(field):
    pattern = r"From (\w+ \d{4}) through current"
    match = re.search(pattern, field)

    if match:
        date_str = match.group(1)
        date_obj = datetime.strptime(date_str, "%B %Y")
        formatted_date_str = date_obj.strftime("%d/%m/%Y")
    else: print(" No date match")
    return formatted_date_str



file_path = 'Sources_Data.csv'                
data = load_csv(file_path)

for row in data[2:]:
    print(f"\\textbf{{{row[0]}}} ", end='', flush=True)
    for field in row[1:len(row)-2]:
        if field != row[1]:
            if field == row[4]:
                date = field.split(' -')[0]
                date = datetime.strptime(date, "%B %d, %Y")
                date = date.strftime("%d/%m/%Y")
                
                print(f" & {date} -{field.split(' -')[1]}", end='', flush=True)
            else:
                print(f" & {field}", end='', flush=True)
    print('\\\\')

for row in data:
    print(f"\\textbf{{{row[0]}}} ", end='', flush=True)
    print(f" & {row[7]}\\\\")
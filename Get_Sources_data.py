import csv

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


file_path = 'Sources_Data.csv'                
data = load_csv(file_path)

for row in data:
    print(f"textbf{{{row[0]}}} ", end='', flush=True)
    for field in row[1:]:
        print(f" & {field}", end='', flush=True)
    print('\\\\')
#print(f"\\textbf{{{source_name}}} & {count}\\\\")
import csv

def read_csv_and_print(filename):
    with open(filename, 'r', newline='') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            print(f"publication({row[0]}) OR ", end='')

# Assuming the csv file is named News_Source_Names.csv
csv_filename = 'News_Source_Names.csv'
read_csv_and_print(csv_filename)

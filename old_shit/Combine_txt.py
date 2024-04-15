"""
Title: Combine Txt
Author: Sean Dowling
Date: 15/04/2024

Description:
This script combines news article data stored as .txt files into one large .txt file.

Inputs:
- News article .txt files (stored in Articles_txt)

Outputs:
- Combined article .txt file (stored as Raw_Articles/Articles_combined.txt)

Dependencies:
- os

"""

import os

def combine_txt_files_in_folder(output_file, folder_path):
    try:
        with open(output_file, 'w', encoding='utf-8', errors='ignore') as output:
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as input_file:
                        content = input_file.read()
                        output.write(content)
                        output.write('\n')
        print(f"Files in '{folder_path}' combined successfully. Result saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
combine_txt_files_in_folder('Raw_Articles/Articles_combined.txt', 'Articles_txt')
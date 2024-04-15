"""
Title: Convert pdf to Txt
Author: Sean Dowling
Date: 15/04/2024

Description:
This script converts news article data stored as .pdf files into .txt files.

Inputs:
- News article .pdf files (stored in Articles_pdf)

Outputs:
- News article .txt files (stored in Articles_txt)

Dependencies:
- os
- fitz

"""

import os
import fitz

# Convert the PDFs to TXT
def convert_pdf_to_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ''
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text += page.get_text()
        return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# The input and output folders
pdf_folder = 'Articles_pdf'
txt_folder = 'Articles_txt'

# Iterate through each PDF file in the input folder
for pdf_file_name in os.listdir(pdf_folder):
    if pdf_file_name.endswith('.PDF'):
        pdf_file_path = os.path.join(pdf_folder, pdf_file_name)
        converted_text = convert_pdf_to_text(pdf_file_path)

        if converted_text:
            # Create the output TXT file path in the output folder
            txt_file_path = os.path.join(txt_folder, pdf_file_name.replace('.PDF', '.txt'))
            
            # Save the text content to the TXT file
            with open(txt_file_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(converted_text)

            print(f'Text content from "{pdf_file_path}" has been saved to "{txt_file_path}"')
        else:
            print(f'Failed to convert PDF to text for "{pdf_file_path}"')
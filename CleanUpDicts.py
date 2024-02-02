import pandas as pd

def remove_duplicates_and_non_letters(input_file, output_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)#
    print(df.head())

    # Remove non-letter characters from all string columns
    df_no_duplicates = df.applymap(lambda x: ''.join(filter(str.isalpha, str(x))) if pd.notna(x) else x)
    
    # Remove duplicates based on all columns
    df_no_duplicates = df_no_duplicates.drop_duplicates()

    # Save the DataFrame without duplicates and non-letter characters to a new CSV file
    df_no_duplicates.to_csv(output_file, index=False)

    print(f"Duplicates removed and non-letter characters removed. Result saved to {output_file}")

# Replace 'input.csv' and 'output.csv' with your actual file names
remove_duplicates_and_non_letters('positive_dict.csv', 'positive_dict.csv')

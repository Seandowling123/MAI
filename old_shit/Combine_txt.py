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
                        output.write('\n')  # Add a newline between the content of each file
        print(f"Files in '{folder_path}' combined successfully. Result saved to '{output_file}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
combine_txt_files_in_folder('Articles_txt_combined/Articles_combined.txt', 'Articles_txt')


# Load articles from text file
def load_txt(file_path):
    try:
        with open(file_path, 'r', encoding="utf-8") as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return f"File not found: {file_path}"
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"
    
def get_non_space_index(row):
    gap_found = False
    for i in range(len(row)):
        if row[i] == ' ':
            gap_found =True
        elif gap_found: return i
    
def get_significance(prob):
    if float(prob) < .01:
        return "***"
    elif float(prob) < .05:
        return "**"
    elif float(prob) < .1:
        return "*"
    else: return ""
    
def get_coefs(text):
    rows = text.split("\n")
    print(rows)
    coefs = []
    for row in rows:
        print(get_non_space_index(row))
        values = row.split('  ')
        if get_significance(values[4]) != "":
            values[1] = values[1] + f"\textsuperscript{{{get_significance(values[4])}}}"
        coefs.append(values[1])
    return coefs
    

file_path = "VAR_Results.txt"
result_txt = load_txt(file_path)
coefs = get_coefs(result_txt)

for coef in coefs:
    print(coef)

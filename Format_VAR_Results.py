import re


def get_values(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        numbers_list = []
        content = file.readlines()
        newline_index = content.index('\n')
        
        # Get variables
        for line in content[:newline_index]:
            line = line.replace('−', '-').replace('*', '')
            first_space_index = line.find(' ')
            variable = line[:first_space_index].strip().split('_')[0]
            numbers = [float(f'{float(num):.6f}') for num in line[first_space_index:].split() if num.strip()]
            numbers_list.append([variable]+numbers)
            
        # Get stats
        stats = "".join(content[newline_index:])
        sum_squared_resid = re.search(r"Sum squared resid\s+([0-9]+)", stats).group(1)
        adjusted_r_squared = re.search(r"Adjusted R-squared\s+([0-9.]+)", stats).group(1)
    return numbers_list, sum_squared_resid, adjusted_r_squared
    
def get_significance(prob):
    if float(prob) < .01:
        return "\\textsuperscript{***}"
    elif float(prob) < .05:
        return "\\textsuperscript{**}"
    elif float(prob) < .1:
        return "\\textsuperscript{*}"
    else: return ""
    
def significance_idk(values):
    for row in values:
        row[1] = str(row[1])+get_significance(row[4])
    return values

def print_values(values):
    iter = 0
    prev_variable = ''
    for row in values:
        if prev_variable != row[0] and iter != 0:
            print("\\midrule")
            #print("")
            iter = 0
        iter = iter+1
        if row[0] == "const":
            print(f"\\textbf{{{row[0]}}} & {row[1]} &")
            #print(f" {row[1]} &")
        else: 
            print(f"\\textbf{{{row[0]}\\textsubscript{{t-{iter}}}}} & {row[1]} \\\\")
            #print(f" {row[1]} &")
        prev_variable = row[0]
    print("\\midrule")
    #print("")
    
file_path = "VAR_Results.txt"
values, sum_squared_resid, adjusted_r_squared = get_values(file_path)
values = significance_idk(values)
print_values(values)
print(f"{sum_squared_resid} & \n{adjusted_r_squared} & ")

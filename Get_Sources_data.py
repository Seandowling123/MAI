

with open("Sources&Counts.txt", 'r', encoding='utf-8') as file:
        content = file.readlines()
        newline_index = content.index('\n')
        
        # Get original article data
        print("Original Articles")
        for line in content[:newline_index]:
            if ': ' in line:
                source_name  = line.split(': ')[0]
                count  = line.split(': ')[1].replace("\n", '')
                #print(f"\\textbf{{{source_name}}} & {count}\\\\")
                print(count)
            
        print("\nFiltered Articles")
        # Get filtered article data
        for line in content[newline_index+1:]:
            if ': ' in line:
                source_name  = line.split(': ')[0]
                count  = line.split(': ')[1].replace("\n", '')
                #print(f"\\textbf{{{source_name}}} & {count}\\\\")
                print(count)
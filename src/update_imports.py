import os
import re

def update_imports_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expressions to match and replace import statements
    updated_content = re.sub(r'from cirkit\.', 'from Cirkits.cirkit.', content)
    updated_content = re.sub(r'import cirkit\.', 'import Cirkits.cirkit.', updated_content)

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(updated_content)

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                update_imports_in_file(file_path)

# Specify the root directory to start the search
root_directory = 'Cirkits/'  # Change this to your target directory

process_directory(root_directory)


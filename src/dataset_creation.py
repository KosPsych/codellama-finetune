import ast
import os
import javalang
import random 
import argparse
from constants import *
 
def extract_python_methods_from_file(file_path):
    """
    Extracts Python methods from a given file.

    Parameters:
    - file_path (str): Path to the Python file.

    Returns:
    - List[str]: List of method names.
    """

    try:
        # Open and read the file content
        with open(file_path, 'r') as file:
            code = file.read()

        # Parse the code into an abstract syntax tree
        tree = ast.parse(code)

    except:
        # If there's an error (e.g., file doesn't exist, parsing error), return an empty list
        return []

    methods = []

    # Traverse the AST and extract method names
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for sub_node in node.body:
                if isinstance(sub_node, ast.FunctionDef):
                    # Combine the class name and method name to get the full method name
                    methods.append(ast.get_source_segment(code, sub_node))

    return methods




def extract_first_method(code):
    """
    Extracts the first method (along with its body) from a given code.

    Parameters:
    - code (str): The source code.

    Returns:
    - str: The extracted method code including its body.
    """

    method_start = None
    open_braces_count = 0
    inside_method = False

    # Iterate over each character in the code
    for idx, char in enumerate(code):
        if char == '{':
            if not inside_method:
                # Mark the start of the method when encountering the first opening brace '{'
                method_start = idx
                inside_method = True
            open_braces_count += 1
        elif char == '}':
            # Decrement the open brace count when encountering a closing brace '}'
            open_braces_count -= 1
            if open_braces_count == 0 and inside_method:
                # If the open brace count returns to zero, we've reached the end of the method
                # Return the method code, including the closing brace '}'
                return code[:idx + 1]

    # If no method is found, return an empty string
    return ""



def extract_java_methods_from_file(file_path):
    """
    Extracts Java methods from a given Java file.

    Parameters:
    - file_path (str): Path to the Java file.

    Returns:
    - List[str]: List of extracted method codes.
    """

    try:
        # Open and read the Java file content
        with open(file_path, 'r') as file:
            code = file.read()

        # Parse the Java code into an abstract syntax tree
        tree = javalang.parse.parse(code)

        methods = []

        # Extract classes and their methods
        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            for _, method_node in node.filter(javalang.tree.MethodDeclaration):
                # Get the starting line of the method
                start_line = method_node.position[0]

                # Split the code by lines and select lines after the method's starting line
                lines = code.split("\n")
                selected_lines = "\n".join(lines[start_line - 1:])  # Adjusted to start from the correct line

                # Extract the first method from the selected lines
                method = extract_first_method(selected_lines)
                methods.append(method)

        return methods

    except:
        # If there's an error (e.g., file doesn't exist, parsing error), return an empty list
        return []








def extract_methods_from_repository(repo_path, dataset_path, n_files):
    """
    Extracts methods from Python and Java files within a repository and saves them to respective files.

    Parameters:
    - repo_path (str): Path to the repository directory.
    - n_files (int): Number of files to process for eacg language

    Returns:
    - None
    """
    
    python_methods = []
    java_methods = []

    python_file_counter = 0
    java_file_counter = 0

    # Walk through the directory structure of the repository
    for subdir, _, files in os.walk(repo_path):
        
        for file in files:
            if file.endswith('.py') and python_file_counter < n_files:
                # Extract Python methods from the file
                
                filepath = os.path.join(subdir, file)
                methods = extract_python_methods_from_file(filepath)

                if random.random()>0.7:
                    sfile = 'test'
                else:
                    sfile = 'train'

                # Save extracted Python methods to a file
                filename = dataset_path + "/python_"+ sfile +".txt"
                    
                   
                with open(filename, "a") as file:
                    for item in methods:
                        file.write(item + "\n###END###\n")
                python_file_counter += 1

            elif file.endswith('.java') and java_file_counter < n_files:
                # Extract Java methods from the file
                filepath = os.path.join(subdir, file)
                methods = extract_java_methods_from_file(filepath)
                # Save extracted Java methods to a file


                if random.random()>0.7:
                    sfile = 'test'
                else:
                    sfile = 'train'
                
                filename = dataset_path+ "/java_" + sfile + ".txt"

                with open(filename, "a") as file:

                    for item in methods:
                    
                        if item !='':   
                            file.write(item + "\n###END###\n")
                        

                java_file_counter  += 1
            
            if java_file_counter  == n_files and python_file_counter  == n_files:
                    # Return early after processing all Java, Python files
                    return






if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Argument parser for dataset extraction")
    
    # Adding positional argument
    parser.add_argument("n_files", type=int, help="Number of files to process for each language")
    args = parser.parse_args()
    n_files = args.n_files
    extract_methods_from_repository(repository_path, dataset_path,  n_files)

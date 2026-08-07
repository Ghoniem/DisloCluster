import subprocess
import os
import pathlib
import sys
import numpy as np
import string, os, math, sys

#Generate file path from directory and file name
def generate_file_paths(directory, names):
    file_paths = []
    for name in names:
        file_path = os.path.join(directory, name)
        file_paths.append(file_path)
    return file_paths


# Get variable from a txt file
def get_scalar(file_path, variable_name):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.split(';', 1)[0]  # Ignore everything after the first semicolon
                parts = line.strip().split('=')
                if len(parts) == 2:
                    name, value = parts
                    if name.strip() == variable_name:
                        print(f"Found variable {variable_name} in the line: {line.strip()}")
                        return float(value.strip())
        return None  # Variable not found in the file
    except FileNotFoundError:
        return None  # File not found

# Get vector from a txt file
def get_vector(file_path, variable_name):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.split(';', 1)[0]  # Ignore everything after the first semicolon
                parts = line.strip().split('=')
                if len(parts) == 2:
                    name, value = parts
                    if name.strip() == variable_name:
                        print(f"Found variable {variable_name} in the line: {line.strip()}")
                        try:
                            vector_str = value.strip()
                            # Add commas between values
                            vector_str_with_commas = ','.join(vector_str.split())
                            vector = np.array([float(item) for item in vector_str_with_commas.split(',')])
                            return vector
                        except ValueError as e:
                            print(f"Error converting to a NumPy array: {e}")
                            return None
        return None  # Variable not found in the file
    except FileNotFoundError:
        return None  # File not found
        
def is_number(s):
    """ Helper function to check if a string can be converted to a float. """
    try:
        float(s)
        return True
    except ValueError:
        return False

def get_matrix(file_path, variable_name):
    try:
        with open(file_path, 'r') as file:
            found_variable = False
            matrix_lines = []
            for line in file:
                clean_line = line.split(';', 1)[0].strip()  # Ignore everything after the first semicolon and strip whitespace
                if not clean_line:
                    continue  # Skip empty lines

                parts = clean_line.split('=')
                if len(parts) == 2:
                    name, value = parts
                    if name.strip() == variable_name:
                        print(f"Found variable {variable_name} in the line: {line.strip()}")
                        found_variable = True
                        matrix_lines.append(value.strip())
                elif found_variable:
                    # Continue collecting lines until a semicolon is encountered
                    if ';' in line:
                        matrix_lines.append(line.split(';')[0].strip())
                        break  # Stop reading when a semicolon is encountered
                    else:
                        matrix_lines.append(clean_line)

            if matrix_lines:
                try:
                    # Join all lines into a single string and split into individual elements
                    matrix_str = ' '.join(matrix_lines)
                    matrix_values = [item for item in matrix_str.split() if is_number(item)]
                    matrix_values = [float(item) for item in matrix_values]
                    # Determine the number of rows (number of lines collected)
                    num_rows = len(matrix_lines)
                    # Calculate the number of columns from the first row
                    num_cols = len(matrix_values) // num_rows
                    # Reshape the matrix
                    matrix = np.array(matrix_values).reshape((num_rows, num_cols))
                    return matrix
                except ValueError as e:
                    print(f"Error converting to a NumPy array: {e}")
                    return None
        return None  # Variable not found in the file
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None  # File not found
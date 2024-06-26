from pathlib import Path
import os


def generate_new_file_path(file_path: Path):
    """
    Generate a new file path by adding a numeric suffix if the file already exists.
    
    Parameters:
    file_path (str): The original file path.
    
    Returns:
    str: A new file path with a numeric suffix if the original file already exists.
    """
    directory, base_name = os.path.split(file_path)
    file_name, extension = os.path.splitext(base_name)
    
    new_file_path = file_path
    counter = 1
    
    while os.path.exists(new_file_path):
        new_file_name = f"{file_name}{counter}{extension}"
        new_file_path = os.path.join(directory, new_file_name)
        counter += 1
    
    return new_file_path
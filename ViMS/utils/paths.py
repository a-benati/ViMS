#!/usr/bin/env python3

import os

def get_working_directory():
    """
    Determine the working directory dynamically.
    """
    return os.getcwd()  # Assumes the script is run from the working directory

def setup_output_dirs():
    """
    Ensure the OUTPUT directory and its subdirectories exist.
    """
    base_output_dir = os.path.join(get_working_directory(), "OUTPUT")
    subdirs = ["LOGS", "CAL_TABLES", "PLOTS"]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_output_dir, subdir), exist_ok=True)
    
    return base_output_dir
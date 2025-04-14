#!/usr/bin/env python3

import os

def get_working_directory():
    """
    Determine the working directory dynamically.
    """
    return os.getcwd()  # Assumes the script is run from the working directory

def setup_output_dirs(obs_id):
    """
    Set up the OUTPUT/<obs_id> directory and its subdirectories.
    
    Parameters:
        obs_id (str): Identifier for the observation (e.g., "obs01").
    
    Returns:
        str: Path to the base output directory for the given observation.
    """
    base_output_dir = os.path.join(get_working_directory(), "OUTPUT", obs_id)
    subdirs = ["LOGS", "CAL_TABLES", "PLOTS"]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_output_dir, subdir), exist_ok=True)
    
    return base_output_dir
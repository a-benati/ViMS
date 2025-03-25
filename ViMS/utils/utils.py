#!/usr/bin/env python3

import subprocess
from ViMS.utils.log import get_logger,log_error
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

# Initialize logger
logger = get_logger(__name__)

def run_command(command):
    """
    Run a shell command and log its output.

    parameters:
        command: Command to execute as a string.
    return:
        Tuple (stdout, stderr) from the command execution.
    """
    try:
        logger.info(f"Executing: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        log_error(logger, f"Command failed: {command}\n {e.stderr.strip() if e.stderr else 'No error output'}")
        raise RuntimeError(f"Command execution failed: {command}")

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

# Define the scope for Google Sheets API access
SCOPE = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

def authenticate_google_api(credentials_file='credentials.json'):
    """
    Authenticate Google API client.
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, SCOPE)
    client = gspread.authorize(credentials)
    return client

def initialize_google_sheet(client, sheet_name='ViMS Pipeline Log'):
    """
    Initialize Google Sheet or open an existing one."""

    try:
        # Try to open an existing sheet
        sheet = client.open(sheet_name).sheet1
    except gspread.exceptions.SpreadsheetNotFound:
        # If not found, create a new sheet
        sheet = client.create(sheet_name).sheet1
        # Set headers
        sheet.append_row(["Timestamp", "Step Name", "Status", "Warnings", "Plot Link"])
    return sheet

# Example usage
# client = authenticate_google_api('credentials.json')
# sheet = initialize_google_sheet(client)

if __name__ == "__main__":
    print(f"Output directory: {setup_output_dirs()}")
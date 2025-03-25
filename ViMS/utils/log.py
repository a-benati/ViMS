#!/usr/bin/env python3

import logging
import os
import glob
import subprocess
from datetime import datetime
from googleapiclient.http import MediaFileUpload
from googleapiclient.discovery import build

class Logger:
    """
    Handles uniform logging for the ViMS pipeline.
    """
    def __init__(self):
        # Set up logging directory
        from ViMS.utils.utils import setup_output_dirs
        logs_dir = os.path.join(setup_output_dirs(), "LOGS")
        log_filename = f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
        log_filepath = os.path.join(logs_dir, log_filename)
        
        # Configure logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filepath),
                logging.StreamHandler()  # Print to console as well
            ]
        )
        
        self.logger = logging.getLogger("ViMS")
        self.log_filepath = log_filepath
    
    def get_logger(self):
        return self.logger
    
    def get_log_filepath(self):
        return self.log_filepath

# Create a global logger instance
logger_instance = Logger()
logger = logger_instance.get_logger()
log_filepath = logger_instance.get_log_filepath()

def delete_casa_logs():
    """
    Find and delete CASA log files.
    """
    casa_logs = glob.glob("casa-*.log")
    for log in casa_logs:
        try:
            os.remove(log)
            logger.info(f"Deleted CASA log file: {log}")
        except Exception as e:
            logger.error(f"Failed to delete CASA log file {log}: {e}")

def redirect_casa_log():
    """
    Run CASA and redirect its log output to the pipeline log file.
    """
    with open(log_filepath, "a") as log_file:
        process = subprocess.Popen(["casa"], stdout=log_file, stderr=log_file)
        process.wait()

def handle_casa_log():
    """
    Run CASA with logging redirection and clean up its log files.
    """
    redirect_casa_log()
    delete_casa_logs()

def append_log_entry(sheet, step_name, status, warnings=None, plot_link=None):
    """
    Append a log entry to the Google Sheet.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    row = [timestamp, step_name, status, warnings if warnings else "", plot_link if plot_link else ""]
    sheet.append_row(row)

# Example usage
# append_log_entry(sheet, "Flagging Calibrators", "Success", "No warnings", "http://link_to_plot.png")

def upload_plot_to_drive(credentials_file='credentials.json', plot_path='path/to/plot.png'):
    """
    Upload a plot to Google Drive and return the shareable link.
    """
    credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, SCOPE)
    drive_service = build('drive', 'v3', credentials=credentials)

    file_metadata = {'name': plot_path.split('/')[-1]}
    media = MediaFileUpload(plot_path, mimetype='image/png')
    
    file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    file_id = file.get('id')

    # Make file publicly accessible (or share it with specific email)
    drive_service.permissions().create(
        fileId=file_id,
        body={'role': 'reader', 'type': 'anyone'},
    ).execute()

    file_url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    return file_url

if __name__ == "__main__":
    logger.info("Logging system initialized.")
    delete_casa_logs()


































# import logging
# import os
# from datetime import datetime

# # Define the logs directory
# LOG_DIR = "../../LOGS"
# os.makedirs(LOG_DIR, exist_ok=True)

# # Generate log filename with timestamp
# log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# # Configure logging
# logging.basicConfig(
#     filename=log_filename,
#     filemode="a",
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )

# # Function to get logger
# def get_logger(name):
#     logger = logging.getLogger(name)
#     return logger

# # Function to log errors with a clear separator
# def log_error(logger, message):
#     separator = "#############################################################"
#     logger.error(f"\n{separator}\n ERROR: {message} \n{separator}")

# # Function to redirect CASA logs to the same file
# def redirect_casa_logs(log_filename):
#     casa_logger = logging.getLogger('casatasks')  # CASA logger
#     casa_handler = logging.FileHandler(log_filename)  # Log to the same file
#     casa_handler.setLevel(logging.INFO)  # Set logging level
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     casa_handler.setFormatter(formatter)
#     casa_logger.addHandler(casa_handler)  # Add handler to CASA logger

# # Function to set up logging for the pipeline and CASA
# def setup_logging():
#     log_filename = get_logger(__name__)
#     redirect_casa_logs(log_filename)  # Redirect CASA logs to the same file
#     return log_filename
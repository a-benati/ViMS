#!/usr/bin/env python3

import logging
import os
import glob
import subprocess
from datetime import datetime
from ViMS.utils.paths import setup_output_dirs
from ViMS.utils.google_api_wrapper import check_or_create_doc, append_log_entry

class Logger:
    """
    Handles uniform logging for the ViMS pipeline.
    """
    def __init__(self):
        # Set up logging directory
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

# Function to initialize the Google Doc by calling google_api_wrapper.py
# def initialize_google_doc():
#     """
#     Initializes the Google Doc by calling the google_api_wrapper.py
#     """
#     result = subprocess.run(
#         ['/opt/py37_env/bin/python3.7', 'google_api_wrapper.py'], 
#         capture_output=True, text=True
#     )

#     # Parse the result from google_api_wrapper.py (if it outputs the document ID)
#     doc_id = result.stdout.strip()  # Assuming the ID is printed in the stdout
#     doc_link = f"https://docs.google.com/document/d/{doc_id}/edit"
    
#     print(f"Google Doc link: {doc_link}")
#     return doc_id

# Function to append updates to the Google Doc
def append_to_google_doc(step_name, status, warnings=None, plot_link=None):
    """
    Appends an update to the Google Doc by calling the google_api_wrapper.py
    """
    # Replace None with an empty string for plot_link
    if plot_link is None:
        plot_link = ""
    subprocess.run(
        ['/opt/py37_env/bin/python3.7', 'google_api_wrapper.py', 'append_log', step_name, status, warnings or "", plot_link]
    )

def initialize_google_doc():
    """
    This function is called ONCE at the start of the pipeline to initialize the Google Doc.
    """
    subprocess.run(['/opt/py37_env/bin/python3.7', 'google_api_wrapper.py', 'init_doc'])

if __name__ == "__main__":
    logger.info("Logging system initialized.")

    # Step 2: Call this function from different parts of the pipeline to append log updates
    initialize_google_doc()
    append_to_google_doc("Step 1: Data Loading", "Success", "No warnings", None)
    append_to_google_doc("Step 2: Processing", "Success", "Minor warning: Skipped some files", "https://drive.google.com/file/d/12345/view")
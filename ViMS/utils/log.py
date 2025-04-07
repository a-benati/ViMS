#!/usr/bin/env python3

import logging
import os
import glob
import subprocess
from datetime import datetime
from utils.client_script import append_log, initialize_google_doc, upload_plot

class Logger:
    """
    Handles uniform logging for the ViMS pipeline.
    """
    def __init__(self, logs_dir):
        # Set up logging directory
        os.makedirs(logs_dir, exist_ok=True)
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

# Function to append updates to the Google Doc
def append_to_google_doc(step_name, status, warnings=None, plot_link=None):
    """
    Appends an update to the Google Doc by calling the Flask backend.
    """
    if plot_link is None:
        plot_link = ""
    if warnings is None:
        warnings = ""
    response = append_log(step_name, status, warnings, plot_link)
    print(f"Appended log response: {response}")

# Function to initialize the Google Doc (called once at pipeline start)
def initialize_google_doc_once():
    """
    Initializes the Google Doc at the beginning of the pipeline.
    """
    response = initialize_google_doc()
    print(f"Initialized doc response: {response}")

# if __name__ == "__main__":
#     logger.info("Logging system initialized.")

    # Step 2: Call this function from different parts of the pipeline to append log updates
    # initialize_google_doc_once()
    # append_to_google_doc("Initial Flagging", "Completed", warnings="", plot_link="")
    #append_to_google_doc("Step 1: Data Loading", "Success", "No warnings", None)
    #append_to_google_doc("Step 2: Processing", "Success", "Minor warning: Skipped some files", "https://drive.google.com/file/d/12345/view")
#!/usr/bin/env python3

import logging
import os, io, sys
import glob, time
import subprocess
import tempfile
import contextlib
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
    
def delete_old_casa_logs():
    """
    Find and delete old CASA log files.
    """
    casa_log_pattern = os.getcwd() + "/casa*.log"
    log_files = glob.glob(casa_log_pattern)
    
    for file in log_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete CASA log file {file}: {e}")

def redirect_casa_log(logger, delay=1.0):
    """
    Finds CASA log file, appends its content to the pipeline log via `logger`,
    and then deletes it from the CASA log file, without deleting the file, so CASA keeps writing in it.

    Args:
        logger: The logger instance of the pipeline.
        delay: Seconds to wait before reading the file (in case CASA is still writing).
    """
    casa_log = glob.glob("casa-*.log")[0]
    if not casa_log:
        logger.warning("No CASA log file found.")
        return
    
    try:
        time.sleep(delay)  # wait in case CASA is finishing writing
        with open(casa_log, "r+", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
            if not lines:
                return
            logger.info("-------------------------------------------------------------------------------------")
            for line in lines:
                logger.info(line.strip())
            logger.info("-------------------------------------------------------------------------------------")
            # Return to the top and clear
            f.seek(0)
            f.truncate()
    except Exception as e:
        logger.error(f"Error reading CASA log file: {e}")

def log_obs_header(logger, obs_id):
    """
    Log a clear, formatted header indicating which observation is being processed.

    Parameters:
        logger: Logger instance to write the messages.
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"########################## PROCESSING {obs_id.upper()} ##########################"
    border = "#" * len(obs_str)

    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info(border)
    logger.info(obs_str)
    logger.info(border)
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")

def log_obs_footer(logger, obs_id):
    """
    Log a clear, formatted footer indicating the observation has finished processing.

    Parameters:
        logger: Logger instance to write the messages.
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"############### FINISHED PROCESSING {obs_id.upper()} ###############"
    border = "#" * len(obs_str)

    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info(border)
    logger.info(obs_str)
    logger.info(border)
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("")

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

# Function to upload a plot to Google Drive
def upload_plot_to_drive(plot_name):
    """
    Upload a plot to Google Drive by calling the Flask backend.
    """
    response = upload_plot(plot_name)
    print(f"Appended log response: {response}")
    return response

# Function to initialize the Google Doc (called once at pipeline start)
def initialize_google_docs_once():
    """
    Initializes the Google Doc at the beginning of the pipeline.
    """
    response = initialize_google_doc()
    print(f"Initialized doc response: {response}")

def log_obs_header_google_doc(obs_id):
    """
    Log the start of an observation's processing in the Google Doc.

    Parameters:
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"############# PROCESSING {obs_id.upper()} #############"
    border = "#" * len(obs_str)

    append_to_google_doc("", "", warnings="", plot_link="")  # Empty line
    append_to_google_doc(border, "", warnings="", plot_link="")
    append_to_google_doc(obs_str, "", warnings="", plot_link="")
    append_to_google_doc(border, "", warnings="", plot_link="")
    append_to_google_doc("", "", warnings="", plot_link="")  # Empty line

def log_obs_footer_google_doc(obs_id):
    """
    Log the end of an observation's processing in the Google Doc.

    Parameters:
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"######## FINISHED PROCESSING {obs_id.upper()} ########"
    border = "#" * len(obs_str)

    append_to_google_doc("", "", warnings="", plot_link="")  # Empty line
    append_to_google_doc(border, "", warnings="", plot_link="")
    append_to_google_doc(obs_str, "", warnings="", plot_link="")
    append_to_google_doc(border, "", warnings="", plot_link="")
    append_to_google_doc("", "", warnings="", plot_link="")  # Empty line
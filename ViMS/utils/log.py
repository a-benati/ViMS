#!/usr/bin/env python3

import logging
import os
import glob, time
import subprocess
from datetime import datetime
from utils.client_script import append_log, initialize_google_docs, upload_plot, update_cell
from casatasks import casalog

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
    
def delete_old_logs():
    """
    Find and delete old CASA log files.
    """
    casa_log_pattern = os.getcwd() + "/casa*.log"
    tricolour_log_pattern = os.getcwd() + "/tricolour*.log"
    casa_log_files = glob.glob(casa_log_pattern)
    tricolour_log_files = glob.glob(tricolour_log_pattern)
    
    for file in casa_log_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete CASA log file {file}: {e}")

    for file in tricolour_log_files:
        try:
            os.remove(file)
        except Exception as e:
            print(f"Failed to delete Tricolour log file {file}: {e}")

def set_casa_log(logger, casa_log_file="casa.log"):
    """
    Set the CASA log file.
    
    Args:
        logger: The logger instance of the pipeline.
        casa_log_file: The name of the CASA log file to use.
    """
    try:
        casalog.setlogfile(casa_log_file)
    except Exception as e:
        logger.error(f"Failed to set CASA log file: {e}")

def redirect_casa_log(logger, delay=1.0):
    """
    Finds CASA log file, appends its content to the pipeline log via `logger`,
    and then deletes it from the CASA log file, without deleting the file, so CASA keeps writing in it.

    Args:
        logger: The logger instance of the pipeline.
        delay: Seconds to wait before reading the file (in case CASA is still writing).
    """
    casa_logs = glob.glob("casa*.log")
    if not casa_logs:
        logger.warning("No CASA log file found.")
        return
    else:
        casa_log = casa_logs[0]
    
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
def append_to_google_doc(step_name, status, warnings=None, plot_link=None, doc_name="ViMS Pipeline Log"):
    """
    Appends an update to the Google Doc by calling the Flask backend.
    """
    if plot_link is None:
        plot_link = ""
    if warnings is None:
        warnings = ""
    response = append_log(step_name, status, warnings, plot_link, doc_name)
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
    response = initialize_google_docs()
    print(f"Initialized doc response: {response}")

def log_obs_header_google_doc(obs_id, doc_name="ViMS Pipeline Plots"):
    """
    Log the start of an observation's processing in the Google Doc.

    Parameters:
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"############# PROCESSING {obs_id.upper()} #############"
    border = "#" * len(obs_str)

    append_to_google_doc("", "", warnings="", plot_link="", doc_name=doc_name)  # Empty line
    append_to_google_doc(border, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc(obs_str, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc(border, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc("", "", warnings="", plot_link="", doc_name=doc_name)  # Empty line

def log_obs_footer_google_doc(obs_id, doc_name="ViMS Pipeline Plots"):
    """
    Log the end of an observation's processing in the Google Doc.

    Parameters:
        obs_id (str): Identifier for the observation (e.g., "obs01").
    """
    obs_str = f"######## FINISHED PROCESSING {obs_id.upper()} ########"
    border = "#" * len(obs_str)

    append_to_google_doc("", "", warnings="", plot_link="", doc_name=doc_name)  # Empty line
    append_to_google_doc(border, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc(obs_str, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc(border, "", warnings="", plot_link="", doc_name=doc_name)
    append_to_google_doc("", "", warnings="", plot_link="", doc_name=doc_name)  # Empty line

def update_cell_in_google_doc(obs_id, column_name, content, is_image=False):
    """
    Update a specific cell in the Google Doc.

    Parameters:
        cell (str): The cell to update (e.g., "A1").
        value (str): The value to set in the cell.
    """
    response = update_cell(obs_id, column_name, content, is_image)
    print(f"Updated cell response: {response}")
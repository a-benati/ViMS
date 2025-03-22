#!/usr/bin/env python3

import logging
import os
from datetime import datetime

# Define the logs directory
LOG_DIR = "../../LOGS"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename with timestamp
log_filename = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# Configure logging
logging.basicConfig(
    filename=log_filename,
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# Function to get logger
def get_logger(name):
    logger = logging.getLogger(name)
    return logger

# Function to log errors with a clear separator
def log_error(logger, message):
    separator = "#############################################################"
    logger.error(f"\n{separator}\n ERROR: {message} \n{separator}")

# Function to redirect CASA logs to the same file
def redirect_casa_logs(log_filename):
    casa_logger = logging.getLogger('casatasks')  # CASA logger
    casa_handler = logging.FileHandler(log_filename)  # Log to the same file
    casa_handler.setLevel(logging.INFO)  # Set logging level
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    casa_handler.setFormatter(formatter)
    casa_logger.addHandler(casa_handler)  # Add handler to CASA logger

# Function to set up logging for the pipeline and CASA
def setup_logging():
    log_filename = get_logger(__name__)
    redirect_casa_logs(log_filename)  # Redirect CASA logs to the same file
    return log_filename
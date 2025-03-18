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

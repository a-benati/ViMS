#!/usr/bin/env python3

import os
from utils import paths, log

# Delete old CASA log files
log.delete_old_casa_logs()

from scripts import flag

# Create the output directories
output_dir = paths.setup_output_dirs()
logs_dir = os.path.join(output_dir, "LOGS")
cal_tables_dir = os.path.join(output_dir, "CAL_TABLES")
plots_dir = os.path.join(output_dir, "PLOTS")

# Create a global logger instance
logger_instance = log.Logger(logs_dir)
logger = logger_instance.get_logger()
log_path = logger_instance.get_log_filepath()

# Initialize Google Doc
log.initialize_google_doc_once()

##########################################################
########################## FLAG ##########################
##########################################################
flag.run(logger, log_path)
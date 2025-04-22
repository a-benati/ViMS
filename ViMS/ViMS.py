#!/usr/bin/env python3

import os
from utils import paths, log

# Delete old CASA log files
log.delete_old_casa_logs()

from scripts import flag

# List of observation IDs
obs_ids = ["obs01"]

# Initialize Google Doc
log.initialize_google_doc_once()

for obs_id in obs_ids:
    
    # Set up output directories for this obs
    output_dir = paths.setup_output_dirs(obs_id)
    logs_dir = os.path.join(output_dir, "LOGS")
    cal_tables_dir = os.path.join(output_dir, "CAL_TABLES")
    plots_dir = os.path.join(output_dir, "PLOTS")
    images_dir = os.path.join(output_dir, "CAL_IMAGES")
    #ms_dir = os.path.join(output_dir, "MS_FILES")
    cubes_dir = os.path.join(output_dir, "STOKES_CUBES")
    ionex_dir = os.path.join(output_dir, "IONEX_DATA")
    
    # Create a logger instance for this obs
    logger_instance = log.Logger(logs_dir)
    logger = logger_instance.get_logger()
    log_path = logger_instance.get_log_filepath()

    # Log the obs header
    log.log_obs_header(logger, obs_id)
    log.log_obs_header_google_doc(obs_id)
    
    ##########################################################
    ########################## FLAG ##########################
    ##########################################################
    flag.run(logger, obs_id)
    
    # Log the obs footer
    log.log_obs_footer(logger, obs_id)
    log.log_obs_footer_google_doc(obs_id)
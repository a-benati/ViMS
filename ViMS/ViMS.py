#!/usr/bin/env python3

import os
from utils import paths, log, cal_ms
import argparse

# Delete old CASA log files
log.delete_old_casa_logs()

from scripts import flag, crosscal, im_polcal, selfcal

# List of observation IDs
OBS_ALL = [f"obs{str(i).zfill(2)}" for i in range(1, 65)]

parser = argparse.ArgumentParser(description="Victoria MeerKAT Survey (ViMS) pipeline.")
parser.add_argument("--obs-id", nargs="+", help="List of observation IDs to run (e.g., obs01 obs02)")
parser.add_argument("--start-from", type=str, help="Start from this observation ID and run all the following ones")
parser.add_argument("--start-step", type=str, choices=["flag", "crosscal", "im_polcal", "selfcal"], default="flag",
                    help="Pipeline step to start from (default: flag)")
args = parser.parse_args()

# Determine list of obs to process
if args.obs_id:
    obs_ids = args.obs_id
elif args.start_from:
    try:
        start_index = OBS_ALL.index(args.start_from)
        obs_ids = OBS_ALL[start_index:]
    except ValueError:
        raise ValueError(f"Observation ID '{args.start_from}' not in the list of known observations.")
else:
    obs_ids = OBS_ALL  # default: run all

# Determine starting step
steps = {"flag": 1, "crosscal": 2, "im_polcal": 3, "selfcal": 4}
current_step = steps[args.start_step]

# Initialize Google Doc
log.initialize_google_docs_once()

for obs_id in obs_ids:
    
    # Set up output directories for this obs
    output_dir = paths.setup_output_dirs(obs_id)
    logs_dir = os.path.join(output_dir, "LOGS")
    cal_tables_dir = os.path.join(output_dir, "CAL_TABLES")
    plots_dir = os.path.join(output_dir, "PLOTS")
    images_dir = os.path.join(output_dir, "CAL_IMAGES")
    ms_dir = os.path.join(output_dir, "MS_FILES")
    cubes_dir = os.path.join(output_dir, "STOKES_CUBES")
    ionex_dir = os.path.join(output_dir, "IONEX_DATA")
    
    # Create a logger instance for this obs
    logger_instance = log.Logger(logs_dir)
    logger = logger_instance.get_logger()
    log_path = logger_instance.get_log_filepath()

    # Names of the Google Docs
    doc_name_log = "ViMS Pipeline Log"
    doc_name_plots = "ViMS Pipeline Plots"

    # Log the obs header
    log.log_obs_header(logger, obs_id)
    #log.log_obs_header_google_doc(obs_id, doc_name_plots)

    # Split full msfile into calibrator ms file (returns full path as a string)
    # cal_ms_file = cal_ms.split_cal(logger, obs_id, output_dir)
    # cal_ms_file = '/localwork/angelina/meerkat_virgo/Obs26/msdir/obs26_1686670937_sdp_l0-cal.ms'
    cal_ms_file = "/a.benati/lw/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms"
    
    ##########################################################
    ########################## FLAG ##########################
    ##########################################################
    if current_step <= 1:
        flag.run(logger, obs_id, cal_ms_file, output_dir)

    ##########################################################
    ######################## CROSSCAL ########################
    ##########################################################
    if current_step <= 2:
        crosscal.run(logger, obs_id, cal_ms_file, output_dir)

    ##########################################################
    ####################### POLCAL IM ########################
    ##########################################################
    if current_step <= 3:
        im_polcal.run(logger, obs_id, cal_ms_file, output_dir)

    ##########################################################
    ######################## SELFCAL #########################
    ##########################################################
    # if current_step <= 4:
    #     selfcal.run(logger, obs_id, cal_ms_file, output_dir)
    
    # Log the obs footer
    log.log_obs_footer(logger, obs_id)
    # log.log_obs_footer_google_doc(obs_id, doc_name_plots)

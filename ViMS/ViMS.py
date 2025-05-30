#!/usr/bin/env python3
from __future__ import annotations
import os
from utils import paths, log, ms_prep
import argparse
import glob

# Delete old CASA log files
log.delete_old_logs()

from scripts import flag, feedswap, crosscal, im_polcal, selfcal

# List of observation IDs
OBS_ALL = [f"obs{str(i).zfill(2)}" for i in range(1, 65)]

parser = argparse.ArgumentParser(description="Victoria MeerKAT Survey (ViMS) pipeline.")
parser.add_argument("--obs-id", nargs="+", help="List of observation IDs to run (e.g., obs01 obs02)")
parser.add_argument("--start-from", type=str, help="Start from this observation ID and run all the following ones")
parser.add_argument("--start-step", type=str, choices=["flag_cal", "crosscal", "im_polcal", "flag_target", "applycal", "selfcal"], default="flag",
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
steps = {"flag_cal": 1, "crosscal": 2, "im_polcal": 3, "flag_target": 4, "applycal": 5, "selfcal": 6}
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
    target_im_dir = os.path.join(output_dir, "TARGET_IMAGES")
    selfcal_dir = os.path.join(output_dir, "SELFCAL_PRODUCTS")
    
    # Create a logger instance for this obs
    logger_instance = log.Logger(logs_dir)
    logger = logger_instance.get_logger()
    log_path = logger_instance.get_log_filepath()

    log.set_casa_log(logger)

    # Names of the Google Docs
    doc_name_log = "ViMS Pipeline Log"
    doc_name_plots = "ViMS Pipeline Plots"

    # Log the obs header
    log.log_obs_header(logger, obs_id)
    #log.log_obs_header_google_doc(obs_id, doc_name_plots)
 
    # copy and unzip the full ms file to the working server /beegfs -NOT YET IMPLEMENTED
    #full_ms = ms_prep.copy_ms(logger, obs_id, output_dir)
    #full_ms = glob.glob(f'/beegfs/bba5268/meerkat_virgo/raw/{obs_id}*sdp_l0.ms')[0]
    full_ms = glob.glob(f'/lofar2/p1uy068/meerkat-virgo/raw/{obs_id}*sdp_l0.ms')[0]
    

    
    ##########################################################
    #################### FLAG CALIBRATORS ####################
    ##########################################################
    if current_step <= 1:
        # Split full msfile into calibrator ms file (returns full path as a string)
        cal_ms_file = ms_prep.split_cal(logger, obs_id, full_ms, output_dir)
        #cal_ms_file = '/localwork/angelina/meerkat_virgo/Obs25/obs25_1686240076_sdp_l0-cal.ms'
        #cal_ms_file = "/a.benati/lw/victoria/tests/flag/obs01_1662797070_sdp_l0-cal_copy.ms"

        # swap the feeds of the calibrator ms file, will skip automatically if already exists
        feedswap.run(logger, cal_ms_file, output_dir)

        flag.run(logger, obs_id, cal_ms_file, output_dir, 'cal')

        #average the calibrator ms file and split into polarisation calibrator and flux/gain claibrator
        pol_ms, flux_ms = ms_prep.average_cal(logger, cal_ms_file, output_dir)

    ##########################################################
    ######################## CROSSCAL ########################
    ##########################################################
    if current_step <= 2:
        crosscal.run(logger, obs_id, flux_ms, pol_ms, output_dir)

    ##########################################################
    ####################### POLCAL IM ########################
    ##########################################################
    if current_step <= 3:
        im_polcal.run(logger, obs_id, pol_ms, output_dir)

    ##########################################################
    ###################### FLAG TARGET #######################
    ##########################################################
    targets = ['virgo064', 'virgo081', 'virgo084', 'virgo101', 'virgo102']
    if current_step <= 4:
        # Split the full ms file into target ms files
        # targets = ['virgo064', 'virgo081', 'virgo084', 'virgo101', 'virgo102']
        # targets = ms_prep.split_targets(logger, obs_id, full_ms, output_dir)

        # swap the feeds of the target ms files, will skip automatically if already exists, then flag them
        for target in targets:
            split_ms = glob.glob(f"{ms_dir}/*{target}.ms")[0]
            feedswap.run(logger, split_ms, output_dir, fields=[0], filename=f"feedswap_{target}.txt")
            flag.run(logger, obs_id, split_ms, output_dir, 'target')

    ##########################################################
    ####################### APPLY CAL ########################
    ##########################################################
    if current_step <= 5:
        #cal_ms.cal_lib(obs_id, logger, "J1939-6342", output_dir)
        new_ms = ms_prep.average_targets(logger, obs_id, targets, output_dir, force=True)
        ms_prep.ionosphere_corr_target(logger, obs_id, targets, new_ms, output_dir)

    ##########################################################
    ######################## SELFCAL #########################
    ##########################################################
    if current_step <= 6:
        selfcal.run(logger, obs_id, targets, output_dir)
    
    # Log the obs footer
    log.log_obs_footer(logger, obs_id)
    # log.log_obs_footer_google_doc(obs_id, doc_name_plots)

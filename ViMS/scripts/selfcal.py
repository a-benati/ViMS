#!/usr/bin/env python3

from utils import utils, log

def run(logger, obs_id, cal_ms, path):
    logger.info("")
    logger.info("")
    logger.info("###########################################################")
    logger.info("######################### SELFCAL #########################")
    logger.info("###########################################################")
    logger.info("")
    logger.info("")

    cmd = f"facetselfcal -h"
    stdout, stderr = utils.run_command(cmd)
    logger.info(stdout)
    if stderr:
        logger.error(f"Error in facetselfcal: {stderr}")





    logger.info("Selfcal step completed successfully!")
    logger.info("")
    logger.info("")
    logger.info("#######################################################")
    logger.info("##################### END SELFCAL #####################")
    logger.info("#######################################################")
    logger.info("")
    logger.info("")
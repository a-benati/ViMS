#!/usr/bin/env python3

import subprocess
from utils.log import get_logger,log_error

# Initialize logger
logger = get_logger(__name__)

def run_command(command):
    """
    Run a shell command and log its output.

    parameters:
        command: Command to execute as a string.
    return:
        Tuple (stdout, stderr) from the command execution.
    """
    try:
        logger.info(f"Executing: {command}")
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        logger.info(result.stdout)
        if result.stderr:
            logger.error(result.stderr)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        log_error(logger, f"Command failed: {command}\n {e.stderr.strip() if e.stderr else 'No error output'}")
        raise RuntimeError(f"Command execution failed: {command}")

#!/usr/bin/env python3

import subprocess
from utils import log

def run_command(command):
    """
    Run a shell command and log its output.

    parameters:
        command: Command to execute as a string.
    return:
        Tuple (stdout, stderr) from the command execution.
    """
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        return result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Command execution failed: {command}")
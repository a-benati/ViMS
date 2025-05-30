#!/usr/bin/env python3

import subprocess
import threading
from utils import log

def run_command(command, logger):
    """
    Run a shell command and log its output.

    parameters:
        command: Command to execute as a string.
    return:
        Tuple (stdout, stderr) from the command execution.
    """
    # try:
    #     result = subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
    #     return result.stdout, result.stderr
    # except subprocess.CalledProcessError as e:
    #     raise RuntimeError(f"Command execution failed: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    stdout_lines = []
    stderr_lines = []

    def read_stream(stream, collect, log_func):
        for line in iter(stream.readline, ''):
            collect.append(line)
            log_func(line.strip())
        stream.close()

    # Thread per leggere stdout e stderr in parallelo
    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, logger.info))
    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, logger.error))

    stdout_thread.start()
    stderr_thread.start()

    process.wait()  # Aspetta la fine del processo principale
    stdout_thread.join()  # Aspetta che i thread di lettura finiscano
    stderr_thread.join()

    return ''.join(stdout_lines), ''.join(stderr_lines)
import sys
import os
import argparse
from pathlib import Path

# Script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add brain age to sys path
ba_path = os.path.join(script_dir, '../')
sys.path.append(os.path.abspath(ba_path))

from autotrainer import ConfigLoader, JobCreator, JobRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to quickly see what examples are parsed as.')
    parser.add_argument('example_file', help='File path to .yml')
    args = parser.parse_args()

    runner_config = ConfigLoader(args.example_file).load()
    job_list = JobCreator(runner_config).create()

    print(job_list)